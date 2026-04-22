// Fixed version - Real scaler values from trained model
// Time mean: 88.35, std: 250.12 | Amount mean: 94813.86, std: 47488.06 (other features unscaled)
const SCALER_MEAN = new Float32Array([88.34961925093133, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,94813.85957508067]);
const SCALER_SCALE = new Float32Array([250.11967013523534, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,47488.062585499334]);

let model = null;
let scaler = null;

document.addEventListener('DOMContentLoaded', async () => {
    initSliders();
    await loadModel();
});

function initSliders() {
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        const valueSpan = document.getElementById(slider.id + 'Value');
        slider.addEventListener('input', () => {
            valueSpan.textContent = slider.value;
            if (slider.id === 'amount') {
                valueSpan.textContent = '$' + parseFloat(slider.value).toLocaleString();
            } else if (slider.id === 'time') {
                valueSpan.textContent = slider.value;
            }
        });
    });
}

async function loadModel() {
    try {
        showLoading(true);
        // Try real model first (add model/ folder later)
        try {
            model = await tf.loadLayersModel('./model/model.json');
            console.log('Real TF.js model loaded successfully');
        } catch (e) {
            console.warn('Real model not found, using robust mock:', e.message);
            model = await createRobustMockModel();
        }
        showLoading(false);
    } catch (error) {
        showError('Model load failed: ' + error.message);
        model = await createRobustMockModel();
        showLoading(false);
    }
}

async function createRobustMockModel() {
    console.log('Using robust mock model');
    // Robust mock NN for demo - simple fraud logic based on dataset patterns
    return {
        predict: async (input) => {
            await tf.nextFrame(); // simulate async
            const data = Array.from(input.dataSync());
            console.log('Mock predict input sample:', data.slice(0,5), data.slice(-3));
            
            const time = data[0];
            const amount = data[29];
            // Simple heuristic mimicking trained model behavior:
            // High amount, unusual time, extreme V features → fraud
            let fraudProb = 0.05; // baseline safe
            if (amount > 0.2 || Math.abs(time) > 2) fraudProb += 0.3; // scaled thresholds
            // Check V features for anomalies (PCA extremes)
            const vAnomaly = data.slice(1,29).some(v => Math.abs(v) > 3);
            if (vAnomaly) fraudProb += 0.4;
            fraudProb = Math.min(0.95, fraudProb + Math.random()*0.1);
            
            const resultTensor = tf.tensor2d([[fraudProb]]);
            console.log('Mock fraud prob:', fraudProb);
            return resultTensor;
        }
    };
}

function scaleFeatures(features) {
    return features.map((val, i) => (val - SCALER_MEAN[i]) / SCALER_SCALE[i]);
}

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const features = new Array(30).fill(0);
    // Time index 0
    const timeEl = document.getElementById('time');
    if (timeEl) features[0] = parseFloat(timeEl.value) || 0;
    // Amount index 29
    const amountEl = document.getElementById('amount');
    if (amountEl) features[29] = parseFloat(amountEl.value) || 0;
    // V1-V28 indices 1-28
    for (let i = 1; i <= 28; i++) {
        const vEl = document.getElementById('v' + i);
        if (vEl) features[i] = parseFloat(vEl.value) || 0;
    }
    
    const scaled = scaleFeatures(features);
    const inputTensor = tf.tensor2d([scaled]);
    
    try {
        console.log('Starting prediction...');
        const prediction = await model.predict(inputTensor);
        console.log('Raw prediction tensor:', prediction);
        
        const probData = await prediction.data();
        const prob = Array.from(probData)[0];
        
        inputTensor.dispose();
        prediction.dispose();
        
        console.log('Final prob:', prob);
        displayResult(prob);
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Prediction failed: ' + error.message);
        if (inputTensor) inputTensor.dispose();
    }
});

function displayResult(prob) {
    const resultDiv = document.getElementById('result');
    const predictionEl = document.getElementById('prediction');
    const fill = document.getElementById('confidenceFill');
    const explanationEl = document.getElementById('explanation');
    
    const isFraud = prob > 0.5;
    const percent = Math.round(prob * 100);
    
    predictionEl.textContent = isFraud ? '🚨 FRAUD DETECTED' : '✅ Transaction Safe';
    predictionEl.style.color = isFraud ? '#e74c3c' : '#27ae60';
    
    fill.style.width = percent + '%';
    
    explanationEl.textContent = `Fraud probability: ${percent}% | ${isFraud ? 'High risk - review transaction' : 'Low risk - approve'}`;
    
    resultDiv.classList.remove('hidden');
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
}

function showError(message) {
    console.error('Error:', message);
    document.getElementById('error').textContent = message;
    document.getElementById('error').classList.remove('hidden');
}

function logFeatures(features) {
    console.log('Input features sample:', features.slice(0,5), '...', features.slice(-3));
}
