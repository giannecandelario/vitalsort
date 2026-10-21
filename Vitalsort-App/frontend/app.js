// ===== APP BOOTSTRAP =====
// All initialization deferred to DOMContentLoaded so every getElementById
// call runs after the HTML is fully parsed. This fixes the ordering bug
// where DOM variables were undefined when the top-block NLP handler registered.
document.addEventListener('DOMContentLoaded', () => {

// 1. DOM Elements
const imageInput = document.getElementById('imageInput');
const uploadImageBtn = document.getElementById('uploadImageBtn');
const visualStatus = document.getElementById('visualStatus');

const nlpInput = document.getElementById('nlpInput');         // Primary visible textarea
const symptomInput = document.getElementById('symptomInput'); // Hidden mirror textarea
const submitNlpBtn = document.getElementById('submitNlpBtn'); // Primary NLP button
const submitTextBtn = document.getElementById('submitTextBtn'); // Secondary NLP button
const nlpStatus = document.getElementById('nlpStatus');

const refreshQueueBtn = document.getElementById('refreshQueueBtn');
const queueStatus = document.getElementById('queueStatus');
const queueBody = document.getElementById('queueBody');

// Contact Form Elements
const submitContactBtn = document.getElementById('submitContactBtn');
const contactStatus = document.getElementById('contactStatus');

// Analytics Elements
const viewAnalyticsBtn = document.getElementById('viewAnalyticsBtn');
const closeAnalyticsBtn = document.getElementById('closeAnalyticsBtn');
const analyticsDashboard = document.getElementById('analytics-dashboard');

// ===== NLP INTAKE HANDLER (submitNlpBtn — primary button) =====
// Now safely registered AFTER DOM elements are declared above.
if (submitNlpBtn) {
    submitNlpBtn.addEventListener('click', async () => {
        const text = nlpInput ? nlpInput.value.trim() : '';

        if (!text) {
            setStatus(nlpStatus, 'error', 'Please enter symptoms before submitting.');
            return;
        }

        setStatus(nlpStatus, 'loading', 'Analyzing...');

        try {
            const response = await fetch('http://localhost:8001/api/intake/nlp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const result = await response.json();

            if (result.is_emergency) {
                setStatus(nlpStatus, 'emergency',
                    `<strong>Emergency Protocol:</strong> ${result.emergency_protocol}<br>Specialist: ${result.specialist}`);
            } else {
                setStatus(nlpStatus, 'success',
                    `<strong>Specialist:</strong> ${result.specialist} &nbsp;·&nbsp; <strong>Severity:</strong> ${result.symptom_severity}`);
            }

            setTimeout(fetchQueue, 1000);

        } catch (error) {
            setStatus(nlpStatus, 'error', 'Request failed: ' + error.message);
        }
    });
}

// 2. WebSocket: Visual Ingestion
uploadImageBtn.addEventListener('click', () => {
    const file = imageInput.files[0];
    if (!file) {
        setStatus(visualStatus, 'error', 'Please select an image before uploading.');
        return;
    }

    setStatus(visualStatus, 'loading', 'Connecting...');
    const ws = new WebSocket('ws://localhost:8001/ws/visual-ingestion');

    ws.onopen = async () => {
        setStatus(visualStatus, 'loading', 'Processing image...');
        try {
            const arrayBuffer = await file.arrayBuffer();
            setStatus(visualStatus, 'loading', 'Uploading to server...');
            ws.send(arrayBuffer);
        } catch (err) {
            setStatus(visualStatus, 'error', err.message);
            ws.close();
        }
    };

    ws.onmessage = (event) => {
        try {
            const response = JSON.parse(event.data);
            if (response.status === 'chunk_received') {
                setStatus(visualStatus, 'success', `Image processed. Record ID: ${response.record_id}`);
                imageInput.value = '';
                fetchQueue();
                ws.close();
            } else if (response.error) {
                setStatus(visualStatus, 'error', response.error);
            }
        } catch (err) {
            setStatus(visualStatus, 'error', 'Parse error: ' + err.message);
        }
    };

    ws.onerror = () => {
        setStatus(visualStatus, 'error', 'WebSocket connection failed.');
    };
});

// 3. REST: NLP Text Intake (secondary handler — submitTextBtn mirrors submitNlpBtn)
if (submitTextBtn) {
    submitTextBtn.addEventListener('click', async () => {
        const textValue = symptomInput.value.trim();
        if (!textValue) {
            setStatus(nlpStatus, 'error', 'Please enter symptoms before submitting.');
            return;
        }

        setStatus(nlpStatus, 'loading', 'Processing...');
        try {
            const response = await fetch('http://localhost:8001/api/intake/nlp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textValue })
            });

            const result = await response.json();
            if (result.status === 'success') {
                if (result.is_emergency) {
                    setStatus(nlpStatus, 'emergency',
                        `<strong>Emergency Protocol:</strong> ${result.emergency_protocol}<br>Specialist: ${result.specialist}`);
                } else {
                    setStatus(nlpStatus, 'success',
                        `<strong>Specialist:</strong> ${result.specialist}`);
                }
                symptomInput.value = '';
                fetchQueue();
            } else {
                setStatus(nlpStatus, 'error', result.error || 'Unknown error');
            }
        } catch (e) {
            setStatus(nlpStatus, 'error', e.message);
        }
    });
}

// Helper — maps priority level text to a badge element
function getPriorityBadge(level) {
    if (!level) return '<span class="badge badge-default">Unknown</span>';
    const l = level.toLowerCase();
    if (l.includes('emergency') || l.includes('immediate') || l.includes('level 1')) {
        return `<span class="badge badge-emergency">${level}</span>`;
    } else if (l.includes('urgent') || l.includes('level 2')) {
        return `<span class="badge badge-urgent">${level}</span>`;
    } else if (l.includes('routine') || l.includes('level 3') || l.includes('level 4') || l.includes('level 5')) {
        return `<span class="badge badge-routine">${level}</span>`;
    }
    return `<span class="badge badge-default">${level}</span>`;
}

// Helper — assigns a status class and optionally sets innerHTML
function setStatus(el, type, html) {
    el.className = 'status status-' + type;
    el.innerHTML = html;
}

// 4. REST: Provider Dashboard
async function fetchQueue() {
    if (queueStatus) queueStatus.textContent = '';
    try {
        const response = await fetch('http://localhost:8001/api/queue');
        const result = await response.json();

        if (result.data.length === 0) {
            queueBody.innerHTML = '<tr class="empty-row"><td colspan="7">No records yet. Submit a triage above to get started.</td></tr>';
        } else {
            queueBody.innerHTML = result.data.map(record => `
                <tr>
                    <td>${record.id}</td>
                    <td>${record.timestamp}</td>
                    <td>${getPriorityBadge(record.priority_level)}</td>
                    <td>${record.symptoms}</td>
                    <td>${record.recommended_action}</td>
                    <td>${record.specialist || '—'}</td>
                    <td><button class="btn-export" onclick="downloadPDF(${record.id})">Export PDF</button></td>
                </tr>
            `).join('');
        }

        if (queueStatus) queueStatus.textContent = `${result.queue_count} record${result.queue_count !== 1 ? 's' : ''}`;
    } catch (e) {
        if (queueStatus) queueStatus.textContent = 'Connection failed';
    }
}

// 5. PDF Export
function downloadPDF(recordId) {
    window.open(`http://localhost:8001/api/export/dossier/${recordId}`, '_blank');
}

// 6. Contact Form Handler
submitContactBtn.addEventListener('click', async () => {
    const name  = document.getElementById('patientName').value.trim();
    const email = document.getElementById('patientEmail').value.trim();
    const phone = document.getElementById('patientPhone').value.trim();
    const notes = document.getElementById('patientNotes').value.trim();

    if (!name || !email || !phone) {
        setStatus(contactStatus, 'error', 'Name, email, and phone are required.');
        return;
    }

    setStatus(contactStatus, 'loading', 'Saving...');

    try {
        const response = await fetch('http://localhost:8001/api/contact/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, email, phone, notes })
        });

        const result = await response.json();
        if (result.status === 'success') {
            setStatus(contactStatus, 'success', 'Contact saved. Follow-up has been scheduled.');
            document.getElementById('contactForm').reset();
        } else {
            setStatus(contactStatus, 'error', result.error || 'Save failed.');
        }
    } catch (e) {
        setStatus(contactStatus, 'error', e.message);
    }
});

// 7. Analytics Dashboard
viewAnalyticsBtn.addEventListener('click', loadAnalyticsDashboard);
closeAnalyticsBtn.addEventListener('click', () => {
    analyticsDashboard.style.display = 'none';
});

async function loadAnalyticsDashboard() {
    analyticsDashboard.style.display = 'block';
    analyticsDashboard.scrollIntoView({ behavior: 'smooth' });
    
    try {
        const response = await fetch('http://localhost:8001/api/analytics/dashboard');
        const data = await response.json();
        
        if (data.status === 'success') {
            displaySystemStats(data.system_stats);
            displayTriageDistribution(data.triage_distribution);
            displaySymptomFrequency(data.symptom_frequency);
            displayDailyTrends(data.acuity_trends);
        }
    } catch (e) {
        alert('Failed to load analytics: ' + e.message);
    }
}

function displaySystemStats(stats) {
    document.getElementById('totalRecords').textContent = stats.total_records;
    document.getElementById('emergencyCount').textContent = stats.emergency_count;
    document.getElementById('emergencyPercent').textContent = stats.emergency_percentage + '%';
    document.getElementById('recordsToday').textContent = stats.records_today;
}

function displayTriageDistribution(data) {
    const dist = data.distribution;
    const counts = Object.values(dist);
    const max = Math.max(...counts, 1);
    let html = '';

    for (const [level, count] of Object.entries(dist)) {
        const pct = Math.round((count / max) * 100);
        const l = level.toLowerCase();
        const barColor = l.includes('emergency') || l.includes('immediate')
            ? 'var(--danger)'
            : l.includes('urgent')
                ? 'var(--warning)'
                : 'var(--brand)';

        html += `
            <div class="dist-item">
                <span class="dist-label">${level}</span>
                <div class="dist-bar-wrap">
                    <div class="dist-bar" style="width:${pct}%; background:${barColor};"></div>
                </div>
                <span class="dist-count">${count}</span>
            </div>
        `;
    }

    document.getElementById('triageChart').innerHTML = html || '<p style="color:var(--text-muted);font-size:13px;">No data available.</p>';
}

function displaySymptomFrequency(data) {
    const symptoms = data.top_symptoms;
    const total = Object.values(symptoms).reduce((a, b) => a + b, 0) || 1;
    let html = '';

    for (const [symptom, count] of Object.entries(symptoms)) {
        const pct = Math.round((count / total) * 100);
        html += `
            <div class="symptom-item">
                <div class="symptom-row">
                    <span class="symptom-name">${symptom}</span>
                    <span class="symptom-count">${count} cases · ${pct}%</span>
                </div>
                <div class="symptom-bar-wrap">
                    <div class="symptom-bar" style="width:${pct}%;"></div>
                </div>
            </div>
        `;
    }

    document.getElementById('symptomsList').innerHTML = html || '<p style="color:var(--text-muted);font-size:13px;">No data available.</p>';
}

function displayDailyTrends(trends) {
    const tbody = document.getElementById('trendsTable');
    let html = '';

    if (trends.daily_trends && trends.daily_trends.length > 0) {
        trends.daily_trends.forEach(day => {
            html += `
                <tr>
                    <td>${day.date}</td>
                    <td style="text-align:center; font-weight:600;">${day.total_cases}</td>
                    <td style="text-align:center; color:var(--danger);">${day.emergencies}</td>
                    <td style="text-align:center; color:var(--warning);">${day.urgent}</td>
                    <td style="text-align:center;">${day.avg_triage_score}</td>
                </tr>
            `;
        });
    } else {
        html = '<tr class="empty-row"><td colspan="5">No trend data available yet.</td></tr>';
    }

    tbody.innerHTML = html;
}

// 8. Event Listeners
refreshQueueBtn.addEventListener('click', fetchQueue);

// 9. Initial queue load on DOM ready
fetchQueue();

console.log("✓ app.js loaded successfully");

}); // END DOMContentLoaded