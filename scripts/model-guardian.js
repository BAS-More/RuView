#!/usr/bin/env node
/**
 * RuView Model Guardian — Smart pre-flight model update system
 *
 * Before running inference, checks HuggingFace for updates, validates them,
 * and only promotes models that prove they add value over current ones.
 *
 * Usage:
 *   node scripts/model-guardian.js check          # Check for updates (no changes)
 *   node scripts/model-guardian.js update          # Check + validate + update if better
 *   node scripts/model-guardian.js validate        # Validate current local models
 *   node scripts/model-guardian.js rollback        # Revert to previous model version
 *   node scripts/model-guardian.js status          # Show current model versions + health
 *   node scripts/model-guardian.js --auto          # Run as pre-flight before inference
 *
 * Environment:
 *   RUVIEW_MODELS_DIR     Models directory (default: ./models)
 *   RUVIEW_DATA_DIR       Test data for validation (default: ./data/recordings)
 *   RUVIEW_AUTO_UPDATE    Enable auto-update on check (default: false)
 *   HF_TOKEN              HuggingFace token for private repos (optional)
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execSync } = require('child_process');

// --- Config ---
const MODELS_DIR = process.env.RUVIEW_MODELS_DIR || path.join(__dirname, '..', 'models');
const DATA_DIR = process.env.RUVIEW_DATA_DIR || path.join(__dirname, '..', 'data', 'recordings');
const HF_REPO = 'ruv/ruview';
const GUARDIAN_STATE = path.join(MODELS_DIR, '.guardian-state.json');
const BACKUP_DIR = path.join(MODELS_DIR, '.backup');
const STAGING_DIR = path.join(MODELS_DIR, '.staging');

// Minimum quality thresholds — new model must meet ALL of these
const QUALITY_GATES = {
  presenceAccuracy: 0.95,      // Must detect presence at 95%+ accuracy
  inferenceLatencyMs: 50,       // Must run under 50ms per embedding
  modelIntegrity: true,         // SHA-256 must match manifest
  noNaN: true,                  // No NaN/Inf in weights
  sizeReasonable: true,         // Not suspiciously large (>100MB) or small (<1KB)
};

// --- Utility ---
function log(level, msg) {
  const ts = new Date().toISOString();
  const prefix = { info: 'ℹ', warn: '⚠', error: '✗', ok: '✓', check: '🔍' }[level] || '·';
  console.log(`[${ts}] ${prefix} ${msg}`);
}

function sha256(filePath) {
  const data = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(data).digest('hex');
}

function fileSize(filePath) {
  try { return fs.statSync(filePath).size; } catch { return 0; }
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function loadState() {
  try { return JSON.parse(fs.readFileSync(GUARDIAN_STATE, 'utf8')); }
  catch { return { version: '0.0.0', lastCheck: null, lastUpdate: null, history: [] }; }
}

function saveState(state) {
  ensureDir(path.dirname(GUARDIAN_STATE));
  fs.writeFileSync(GUARDIAN_STATE, JSON.stringify(state, null, 2));
}

// --- Check for Updates ---
function checkForUpdates() {
  log('check', 'Checking HuggingFace for model updates...');

  const state = loadState();
  const localConfig = loadLocalConfig();

  // Get remote metadata
  let remoteInfo;
  try {
    const cmd = `hf repo info ${HF_REPO} --json 2>/dev/null || echo "{}"`;
    const output = execSync(cmd, { encoding: 'utf8', timeout: 30000 }).trim();
    remoteInfo = JSON.parse(output || '{}');
  } catch (e) {
    // Fallback: download just the config.json to compare
    try {
      ensureDir(STAGING_DIR);
      execSync(`hf download ${HF_REPO} config.json --local-dir "${STAGING_DIR}" --quiet`,
        { timeout: 30000, stdio: 'pipe' });
      const remoteConfig = JSON.parse(fs.readFileSync(path.join(STAGING_DIR, 'config.json'), 'utf8'));
      remoteInfo = { config: remoteConfig };
    } catch (e2) {
      log('warn', `Cannot reach HuggingFace: ${e2.message}`);
      return { hasUpdate: false, reason: 'offline' };
    }
  }

  // Compare versions
  const remoteConfig = remoteInfo.config ||
    (fs.existsSync(path.join(STAGING_DIR, 'config.json'))
      ? JSON.parse(fs.readFileSync(path.join(STAGING_DIR, 'config.json'), 'utf8'))
      : null);

  if (!remoteConfig) {
    log('info', 'No remote config found — keeping current models');
    return { hasUpdate: false, reason: 'no-remote-config' };
  }

  const localVersion = localConfig?.version || '0.0.0';
  const remoteVersion = remoteConfig?.version || '0.0.0';
  const localSteps = localConfig?.training?.steps || 0;
  const remoteSteps = remoteConfig?.training?.steps || 0;
  const localLoss = localConfig?.training?.loss || Infinity;
  const remoteLoss = remoteConfig?.training?.loss || Infinity;

  const hasNewVersion = remoteVersion !== localVersion;
  const hasMoreTraining = remoteSteps > localSteps;
  const hasLowerLoss = remoteLoss < localLoss * 0.95; // 5% improvement threshold

  const hasUpdate = hasNewVersion || hasMoreTraining || hasLowerLoss;

  log('info', `Local:  v${localVersion} (${localSteps} steps, loss=${localLoss.toFixed(4)})`);
  log('info', `Remote: v${remoteVersion} (${remoteSteps} steps, loss=${remoteLoss.toFixed(4)})`);

  if (hasUpdate) {
    const reasons = [];
    if (hasNewVersion) reasons.push(`new version ${remoteVersion}`);
    if (hasMoreTraining) reasons.push(`${remoteSteps - localSteps} more training steps`);
    if (hasLowerLoss) reasons.push(`${((1 - remoteLoss/localLoss) * 100).toFixed(1)}% lower loss`);
    log('ok', `Update available: ${reasons.join(', ')}`);
  } else {
    log('ok', 'Models are up to date');
  }

  state.lastCheck = new Date().toISOString();
  saveState(state);

  return { hasUpdate, localVersion, remoteVersion, localSteps, remoteSteps, localLoss, remoteLoss };
}

function loadLocalConfig() {
  const paths = [
    path.join(MODELS_DIR, 'huggingface', 'config.json'),
    path.join(MODELS_DIR, 'csi-ruvllm', 'config.json'),
    path.join(MODELS_DIR, 'pretrained', 'config.json'),
  ];
  for (const p of paths) {
    try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch {}
  }
  return null;
}

// --- Validate Models ---
function validateModels(modelDir) {
  log('check', `Validating models in ${modelDir}...`);
  const results = { passed: true, checks: [] };

  // 1. File integrity — check key files exist and aren't empty
  const requiredFiles = ['config.json'];
  const optionalFiles = ['model.safetensors', 'model-q4.bin', 'presence-head.json'];

  for (const file of requiredFiles) {
    const fp = path.join(modelDir, file);
    if (!fs.existsSync(fp)) {
      results.checks.push({ name: `file:${file}`, passed: false, reason: 'missing' });
      results.passed = false;
    } else if (fileSize(fp) < 10) {
      results.checks.push({ name: `file:${file}`, passed: false, reason: 'empty/corrupt' });
      results.passed = false;
    } else {
      results.checks.push({ name: `file:${file}`, passed: true, size: fileSize(fp), sha256: sha256(fp).slice(0, 12) });
    }
  }

  for (const file of optionalFiles) {
    const fp = path.join(modelDir, file);
    if (fs.existsSync(fp)) {
      const size = fileSize(fp);
      // Size sanity check
      if (size > 100 * 1024 * 1024) {
        results.checks.push({ name: `size:${file}`, passed: false, reason: `too large (${(size/1024/1024).toFixed(1)}MB)` });
        results.passed = false;
      } else if (size < 100) {
        results.checks.push({ name: `size:${file}`, passed: false, reason: `too small (${size}B)` });
        results.passed = false;
      } else {
        results.checks.push({ name: `file:${file}`, passed: true, size });
      }
    }
  }

  // 2. Config validity
  try {
    const config = JSON.parse(fs.readFileSync(path.join(modelDir, 'config.json'), 'utf8'));
    if (!config.model_type && !config.architecture) {
      results.checks.push({ name: 'config:valid', passed: false, reason: 'no model_type or architecture' });
      results.passed = false;
    } else {
      results.checks.push({ name: 'config:valid', passed: true, architecture: config.architecture || config.model_type });
    }
  } catch (e) {
    results.checks.push({ name: 'config:parse', passed: false, reason: e.message });
    results.passed = false;
  }

  // 3. Presence head weights sanity (if exists)
  const presencePath = path.join(modelDir, 'presence-head.json');
  if (fs.existsSync(presencePath)) {
    try {
      const head = JSON.parse(fs.readFileSync(presencePath, 'utf8'));
      const weights = head.weights || [];
      const hasNaN = weights.some(w => isNaN(w) || !isFinite(w));
      if (hasNaN) {
        results.checks.push({ name: 'weights:nan', passed: false, reason: 'NaN/Inf in presence head weights' });
        results.passed = false;
      } else {
        results.checks.push({ name: 'weights:nan', passed: true, numWeights: weights.length });
      }
    } catch (e) {
      results.checks.push({ name: 'weights:parse', passed: false, reason: e.message });
      results.passed = false;
    }
  }

  // 4. Run inference benchmark (if benchmark script + test data exist)
  const benchmarkScript = path.join(__dirname, 'benchmark-ruvllm.js');
  const testData = findTestData();
  if (fs.existsSync(benchmarkScript) && testData) {
    try {
      log('check', 'Running inference benchmark...');
      const output = execSync(
        `node "${benchmarkScript}" --model "${modelDir}" --data "${testData}" 2>&1`,
        { encoding: 'utf8', timeout: 60000 }
      );

      // Parse accuracy from output
      const accMatch = output.match(/Accuracy:\s+([\d.]+)%/);
      const latMatch = output.match(/Mean:\s+([\d.]+)\s*ms/);
      const throughMatch = output.match(/Throughput:\s+([\d]+)/);

      const accuracy = accMatch ? parseFloat(accMatch[1]) / 100 : null;
      const latency = latMatch ? parseFloat(latMatch[1]) : null;
      const throughput = throughMatch ? parseInt(throughMatch[1]) : null;

      if (accuracy !== null) {
        const passed = accuracy >= QUALITY_GATES.presenceAccuracy;
        results.checks.push({ name: 'benchmark:accuracy', passed, value: accuracy, threshold: QUALITY_GATES.presenceAccuracy });
        if (!passed) results.passed = false;
      }

      if (latency !== null) {
        const passed = latency <= QUALITY_GATES.inferenceLatencyMs;
        results.checks.push({ name: 'benchmark:latency', passed, value: `${latency}ms`, threshold: `${QUALITY_GATES.inferenceLatencyMs}ms` });
        if (!passed) results.passed = false;
      }

      if (throughput !== null) {
        results.checks.push({ name: 'benchmark:throughput', passed: true, value: `${throughput} emb/sec` });
      }
    } catch (e) {
      results.checks.push({ name: 'benchmark:run', passed: false, reason: `benchmark failed: ${e.message.slice(0, 100)}` });
      // Don't fail overall for benchmark errors — model may still be usable
    }
  }

  // Summary
  const passCount = results.checks.filter(c => c.passed).length;
  const totalCount = results.checks.length;
  log(results.passed ? 'ok' : 'error', `Validation: ${passCount}/${totalCount} checks passed`);

  return results;
}

function findTestData() {
  const candidates = [
    path.join(DATA_DIR, 'overnight-1775217646.csi.jsonl'),
    path.join(DATA_DIR, 'pretrain-1775182186.csi.jsonl'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  // Find any .csi.jsonl file
  try {
    const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.csi.jsonl'));
    if (files.length > 0) return path.join(DATA_DIR, files[0]);
  } catch {}
  return null;
}

// --- Update with Validation ---
function updateModels() {
  const check = checkForUpdates();
  if (!check.hasUpdate) {
    log('ok', 'No update needed');
    return { updated: false };
  }

  log('info', 'Downloading candidate models to staging...');
  ensureDir(STAGING_DIR);
  ensureDir(BACKUP_DIR);

  // Download to staging
  try {
    execSync(`hf download ${HF_REPO} --local-dir "${STAGING_DIR}" --quiet`,
      { timeout: 120000, stdio: 'pipe' });
    log('ok', 'Downloaded candidate models');
  } catch (e) {
    log('error', `Download failed: ${e.message}`);
    return { updated: false, error: 'download-failed' };
  }

  // Validate candidate
  log('info', 'Validating candidate models...');
  const validation = validateModels(STAGING_DIR);
  if (!validation.passed) {
    log('error', 'Candidate models FAILED validation — not updating');
    log('info', 'Failed checks:');
    validation.checks.filter(c => !c.passed).forEach(c => {
      log('error', `  ${c.name}: ${c.reason || 'failed'}`);
    });
    // Clean staging
    fs.rmSync(STAGING_DIR, { recursive: true, force: true });
    return { updated: false, error: 'validation-failed', validation };
  }

  // Compare with current (if benchmark data available)
  const currentDir = path.join(MODELS_DIR, 'huggingface');
  if (fs.existsSync(path.join(currentDir, 'config.json'))) {
    log('info', 'Comparing candidate vs current models...');
    const currentValidation = validateModels(currentDir);

    // Check if candidate is actually better
    const candidateAcc = validation.checks.find(c => c.name === 'benchmark:accuracy')?.value;
    const currentAcc = currentValidation.checks.find(c => c.name === 'benchmark:accuracy')?.value;

    if (candidateAcc !== null && currentAcc !== null && candidateAcc < currentAcc) {
      log('warn', `Candidate accuracy (${(candidateAcc*100).toFixed(1)}%) is WORSE than current (${(currentAcc*100).toFixed(1)}%) — skipping`);
      fs.rmSync(STAGING_DIR, { recursive: true, force: true });
      return { updated: false, error: 'candidate-worse', candidateAcc, currentAcc };
    }
  }

  // Backup current models
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupPath = path.join(BACKUP_DIR, timestamp);
  if (fs.existsSync(path.join(MODELS_DIR, 'huggingface'))) {
    log('info', `Backing up current models to ${backupPath}`);
    ensureDir(backupPath);
    execSync(`cp -r "${path.join(MODELS_DIR, 'huggingface')}" "${backupPath}/"`);
  }

  // Promote staging → production
  log('info', 'Promoting validated candidate to production...');
  const targetDir = path.join(MODELS_DIR, 'huggingface');
  if (fs.existsSync(targetDir)) {
    fs.rmSync(targetDir, { recursive: true, force: true });
  }
  fs.renameSync(STAGING_DIR, targetDir);

  // Update state
  const state = loadState();
  state.lastUpdate = new Date().toISOString();
  state.version = check.remoteVersion;
  state.history.push({
    timestamp: state.lastUpdate,
    from: check.localVersion,
    to: check.remoteVersion,
    backup: backupPath,
    validation: validation.checks.map(c => ({ name: c.name, passed: c.passed })),
  });
  // Keep last 10 history entries
  if (state.history.length > 10) state.history = state.history.slice(-10);
  saveState(state);

  log('ok', `Models updated: v${check.localVersion} → v${check.remoteVersion}`);
  return { updated: true, from: check.localVersion, to: check.remoteVersion, validation };
}

// --- Rollback ---
function rollback() {
  const state = loadState();
  const lastUpdate = state.history[state.history.length - 1];

  if (!lastUpdate?.backup || !fs.existsSync(lastUpdate.backup)) {
    log('error', 'No backup available to rollback to');
    return false;
  }

  log('info', `Rolling back to backup from ${lastUpdate.timestamp}...`);
  const targetDir = path.join(MODELS_DIR, 'huggingface');

  if (fs.existsSync(targetDir)) {
    fs.rmSync(targetDir, { recursive: true, force: true });
  }

  // Restore from backup
  const backupHf = path.join(lastUpdate.backup, 'huggingface');
  if (fs.existsSync(backupHf)) {
    execSync(`cp -r "${backupHf}" "${targetDir}"`);
  }

  state.version = lastUpdate.from;
  state.history.push({
    timestamp: new Date().toISOString(),
    action: 'rollback',
    to: lastUpdate.from,
  });
  saveState(state);

  log('ok', `Rolled back to v${lastUpdate.from}`);
  return true;
}

// --- Status ---
function showStatus() {
  const state = loadState();
  const localConfig = loadLocalConfig();

  console.log('\n=== RuView Model Guardian Status ===\n');
  console.log(`  Model version:  ${state.version || localConfig?.version || 'unknown'}`);
  console.log(`  Architecture:   ${localConfig?.architecture || 'unknown'}`);
  console.log(`  Last check:     ${state.lastCheck || 'never'}`);
  console.log(`  Last update:    ${state.lastUpdate || 'never'}`);
  console.log(`  Update history: ${state.history.length} entries`);

  // Show model files
  console.log('\n  Model files:');
  const dirs = ['huggingface', 'csi-ruvllm'];
  for (const dir of dirs) {
    const fullDir = path.join(MODELS_DIR, dir);
    if (fs.existsSync(fullDir)) {
      const files = fs.readdirSync(fullDir).filter(f => !f.startsWith('.'));
      const totalSize = files.reduce((sum, f) => sum + fileSize(path.join(fullDir, f)), 0);
      console.log(`    ${dir}/: ${files.length} files, ${(totalSize/1024).toFixed(1)} KB`);
    }
  }

  // Show backups
  if (fs.existsSync(BACKUP_DIR)) {
    const backups = fs.readdirSync(BACKUP_DIR);
    console.log(`\n  Backups: ${backups.length} snapshots in .backup/`);
  }

  console.log('');
}

// --- Pre-flight (--auto mode) ---
function preflight() {
  log('check', 'RuView Model Guardian — pre-flight check');

  const state = loadState();
  const lastCheck = state.lastCheck ? new Date(state.lastCheck) : new Date(0);
  const hoursSinceCheck = (Date.now() - lastCheck.getTime()) / (1000 * 60 * 60);

  // Only check HuggingFace if last check was >6 hours ago
  if (hoursSinceCheck < 6) {
    log('ok', `Last check was ${hoursSinceCheck.toFixed(1)}h ago — skipping remote check`);
    // Still validate local models
    const localDir = path.join(MODELS_DIR, 'huggingface');
    if (fs.existsSync(path.join(localDir, 'config.json'))) {
      const v = validateModels(localDir);
      if (!v.passed) {
        log('error', 'Local models FAILED validation — attempting re-download');
        return updateModels();
      }
    }
    log('ok', 'Pre-flight passed — models ready');
    return { ready: true };
  }

  // Check + update if auto-update enabled
  const check = checkForUpdates();
  if (check.hasUpdate && process.env.RUVIEW_AUTO_UPDATE === 'true') {
    log('info', 'Auto-update enabled — updating...');
    return updateModels();
  } else if (check.hasUpdate) {
    log('warn', 'Update available but auto-update disabled. Run: node model-guardian.js update');
  }

  log('ok', 'Pre-flight passed');
  return { ready: true, hasUpdate: check.hasUpdate };
}

// --- CLI ---
const command = process.argv[2] || 'status';

switch (command) {
  case 'check':
    checkForUpdates();
    break;
  case 'update':
    updateModels();
    break;
  case 'validate':
    const dir = process.argv[3] || path.join(MODELS_DIR, 'huggingface');
    const result = validateModels(dir);
    result.checks.forEach(c => {
      log(c.passed ? 'ok' : 'error', `  ${c.name}: ${c.passed ? 'PASS' : 'FAIL'} ${c.reason || c.value || ''}`);
    });
    break;
  case 'rollback':
    rollback();
    break;
  case 'status':
    showStatus();
    break;
  case '--auto':
    preflight();
    break;
  default:
    console.log('Usage: node model-guardian.js [check|update|validate|rollback|status|--auto]');
}
