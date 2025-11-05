// static/js/main.js
document.addEventListener('DOMContentLoaded', async () => {

    // --- Range sliders display ---
    function bindRange(id, displayId) {
        const el = document.getElementById(id);
        const disp = document.getElementById(displayId);
        if (!el || !disp) return;
        el.addEventListener('input', () => disp.innerText = el.value);
        disp.innerText = el.value;
    }
    ['Age', 'Income', 'LoanAmount', 'CreditScore', 'InterestRate', 'DTIRatio']
        .forEach(id => bindRange(id, id.toLowerCase() + 'Val'));

    // --- Helper: fill dropdowns ---
    function fillSelect(id, options) {
        const sel = document.getElementById(id);
        if (!sel) return;
        sel.innerHTML = '';
        options.forEach(opt => {
            const o = document.createElement('option');
            o.value = opt;
            o.innerText = opt;
            sel.appendChild(o);
        });
    }

    // --- Fetch dropdown data ---
    try {
        const res = await fetch('/api/visuals');
        if (!res.ok) throw new Error('API /api/visuals not found');
        const data = await res.json();
        const edu = (data.default_by_education || []).map(d => d.Education);
        fillSelect('Education', edu.length ? edu : ['High School', 'Bachelor', 'Master', 'PhD']);
        fillSelect('EmploymentType', ['Salaried', 'Self Employed', 'Unemployed', 'Student']);
        fillSelect('MaritalStatus', ['Single', 'Married', 'Divorced']);
        fillSelect('HasMortgage', ['Yes', 'No']);
        fillSelect('HasDependents', ['Yes', 'No']);
        fillSelect('LoanPurpose', ['Home', 'Car', 'Personal', 'Business', 'Education', 'Other']);
        fillSelect('HasCoSigner', ['Yes', 'No']);
    } catch {
        // fallback
        ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'].forEach(id => {
            switch (id) {
                case 'Education': fillSelect(id, ['High School', 'Bachelor', 'Master', 'PhD']); break;
                case 'EmploymentType': fillSelect(id, ['Salaried', 'Self Employed', 'Unemployed', 'Student']); break;
                case 'MaritalStatus': fillSelect(id, ['Single', 'Married', 'Divorced']); break;
                case 'HasMortgage':
                case 'HasDependents':
                case 'HasCoSigner': fillSelect(id, ['Yes', 'No']); break;
                case 'LoanPurpose': fillSelect(id, ['Home', 'Car', 'Personal', 'Business', 'Education', 'Other']); break;
            }
        });
    }

    // --- Predict loan default ---
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const payload = {};
            new FormData(predictForm).forEach((v, k) => payload[k] = v);

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) throw new Error('Prediction API failed');
                const data = await res.json();

                if (data.error) {
                    document.getElementById('riskText').innerText = "Error: " + data.error;
                    return;
                }

                const prob = data.probability || 0;
                document.getElementById('riskText').innerText = `Default probability: ${Math.round(prob * 100)}%`;
                document.getElementById('suggestion').innerText = data.suggestion || '';
                document.getElementById('clusterInfo').innerText = data.cluster ? `Customer segment: ${data.cluster}` : '';

                // --- Risk Pie Chart ---
                const ctx = document.getElementById('riskPie').getContext('2d');
                if (window.riskPieChart) window.riskPieChart.destroy();
                window.riskPieChart = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Default', 'No default'],
                        datasets: [{ data: [prob, 1 - prob], backgroundColor: ['#dc3545', '#28a745'] }]
                    },
                    options: { plugins: { legend: { position: 'bottom' } } }
                });

                // --- Repayment Chart ---
                const schedule = data.schedule || [];
                document.getElementById('repaySummary').innerHTML = `
                    <p>Estimated months to repay: <b>${data.estimated_months || '-'}</b><br>
                    Monthly payment: <b>${data.monthly_payment || '-'}</b></p>`;

                const repayCtx = document.getElementById('repayChart').getContext('2d');
                const months = schedule.map(s => s.month);
                const outstanding = schedule.map(s => s.outstanding);

                if (window.repayChart) window.repayChart.destroy();
                window.repayChart = new Chart(repayCtx, {
                    type: 'line',
                    data: { labels: months, datasets: [{ label: 'Outstanding balance', data: outstanding, fill: true }] },
                    options: {
                        responsive: true,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { title: { display: true, text: 'Month' } },
                            y: { title: { display: true, text: 'Outstanding Amount' } }
                        }
                    }
                });

            } catch (err) {
                console.error(err);
                document.getElementById('riskText').innerText = "Server error: Unable to predict.";
            }
        });
    }

    // --- Visuals ---
    try {
        const res = await fetch('/api/visuals');
        if (!res.ok) throw new Error('Visuals API not found');
        const d = await res.json();

        const eduData = d.default_by_education || [];
        if (eduData.length) {
            const eduLabels = eduData.map(x => x.Education);
            const eduRates = eduData.map(x => x.rate || x.Default || 0);
            const eduCtx = document.getElementById('eduChart').getContext('2d');
            new Chart(eduCtx, {
                type: 'bar',
                data: { labels: eduLabels, datasets: [{ label: 'Default rate', data: eduRates, backgroundColor: '#007bff' }] },
                options: { plugins: { legend: { display: false } } }
            });
        }

        const loanVals = d.loan_amount_values || [];
        if (loanVals.length) {
            const min = Math.min(...loanVals), max = Math.max(...loanVals);
            const bins = 10, binSize = (max - min) / bins;
            const counts = Array(bins).fill(0);
            const labels = Array.from({ length: bins }, (_, i) => `${Math.round(min + i * binSize)} - ${Math.round(min + (i + 1) * binSize)}`);
            loanVals.forEach(v => counts[Math.min(bins - 1, Math.floor((v - min) / binSize))]++);
            const loanCtx = document.getElementById('loanHist').getContext('2d');
            new Chart(loanCtx, {
                type: 'bar',
                data: { labels, datasets: [{ label: 'Loan count', data: counts, backgroundColor: '#28a745' }] },
                options: { plugins: { legend: { display: false } } }
            });
        }

    } catch (e) {
        console.warn("Visuals load failed:", e);
    }

    // --- Chatbot ---
    const sendChat = document.getElementById('sendChat');
    if (sendChat) {
        sendChat.addEventListener('click', async () => {
            const q = document.getElementById('chatInput').value.trim();
            if (!q) return;
            const box = document.getElementById('chatBox');
            box.innerHTML += `<div><b>You:</b> ${q}</div>`;
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: q })
                });
                const r = await res.json();
                box.innerHTML += `<div><b>Bot:</b> ${r.reply || 'No response'}</div>`;
            } catch {
                box.innerHTML += `<div><b>Bot:</b> Server not responding</div>`;
            }
        });
    }

    // --- Loan Recommender ---
    const recommendBtn = document.getElementById('recommendBtn');
    if (recommendBtn) {
        recommendBtn.addEventListener('click', async () => {
            const form = document.getElementById('recommendForm');
            const payload = {};
            new FormData(form).forEach((v, k) => payload[k] = v);

            try {
                const res = await fetch('/api/recommend_bank', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) throw new Error('API failed');
                const data = await res.json();
                const box = document.getElementById('bankRecommendation');
                const recs = data.recommendations || [];

                if (data.error) {
                    box.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
                } else if (!recs.length) {
                    box.innerHTML = `<p style="color:orange;">${data.message || 'No recommendations available.'}</p>`;
                } else {
                    box.innerHTML = `<h5>🏦 Recommended Banks</h5><ul>` +
                        recs.map(r => `<li><b>${r.Bank_Name}</b> — Interest: ${r.Interest_Rate}% | Fee: ₹${r.Processing_Fee}</li>`).join('') +
                        `</ul>`;
                }

            } catch (err) {
                console.error(err);
                document.getElementById('bankRecommendation').innerHTML = `<p style="color:red;">Server not responding</p>`;
            }
        });
    }
});

// ================================
// 🎨 Customer Segmentation Visuals
// ================================

fetch('/api/clusters')
    .then(res => res.json())
    .then(data => {
        const ctxCluster = document.getElementById('clusterChart');
        const ctxPie = document.getElementById('clusterPie');

        // Scatter plot of clusters
        new Chart(ctxCluster, {
            type: 'scatter',
            data: {
                datasets: data.clusters.map((cluster, i) => ({
                    label: `Cluster ${i + 1}`,
                    data: cluster.points.map(p => ({ x: p.income, y: p.default_prob })),
                    backgroundColor: cluster.color,
                    pointRadius: 5
                }))
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Customer Segmentation by Income & Default Probability'
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Income' } },
                    y: { title: { display: true, text: 'Default Probability' } }
                }
            }
        });

        // Pie chart showing cluster sizes
        new Chart(ctxPie, {
            type: 'pie',
            data: {
                labels: data.clusters.map((_, i) => `Cluster ${i + 1}`),
                datasets: [{
                    data: data.clusters.map(c => c.size),
                    backgroundColor: data.clusters.map(c => c.color)
                }]
            },
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: 'Cluster Size Distribution'
                    }
                }
            }
        });
    })
    .catch(err => console.error("Visuals load failed:", err));

