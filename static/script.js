let debounceTimer;

async function getSuggestions(query) {
    if (!query) {
        document.getElementById('suggestions').style.display = 'none';
        return;
    }

    try {
        const response = await fetch(`/suggest?query=${encodeURIComponent(query)}`);
        const suggestions = await response.json();
        
        const suggestionsContainer = document.getElementById('suggestions');
        
        if (suggestions.length === 0) {
            suggestionsContainer.style.display = 'none';
            return;
        }

        suggestionsContainer.innerHTML = suggestions.map(item => `
            <div class="suggestion-item" onclick="selectStock('${item.symbol}', '${item.name}')">
                <span class="suggestion-symbol">${item.symbol}</span>
                <span class="suggestion-name">${item.name}</span>
            </div>
        `).join('');

        suggestionsContainer.style.display = 'block';
    } catch (err) {
        console.error('Error fetching suggestions:', err);
    }
}

function selectStock(symbol, name) {
    document.getElementById('stockSymbol').value = symbol;
    document.getElementById('suggestions').style.display = 'none';
    predictStock();
}

// Add event listener for input changes
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('stockSymbol');
    const suggestions = document.getElementById('suggestions');

    input.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            getSuggestions(e.target.value.trim());
        }, 300);
    });

    // Close suggestions when clicking outside
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !suggestions.contains(e.target)) {
            suggestions.style.display = 'none';
        }
    });
});

async function predictStock() {
    const stockSymbol = document.getElementById('stockSymbol').value.trim().toUpperCase();
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const error = document.getElementById('error');

    if (!stockSymbol) {
        showError('Please enter a stock symbol');
        return;
    }

    // Show loading and hide other sections
    loading.classList.remove('hidden');
    result.classList.add('hidden');
    error.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symbol: stockSymbol })
        });

        const data = await response.json();

        if (data.error) {
            showError(data.error);
            return;
        }

        // Update current price
        document.getElementById('currentPrice').textContent = `₹${data.current_price}`;

        // Update predictions grid
        const predictionsGrid = document.getElementById('predictionsGrid');
        predictionsGrid.innerHTML = ''; // Clear existing predictions

        data.predictions.forEach(pred => {
            const card = document.createElement('div');
            card.className = 'prediction-card';
            
            const dayText = pred.day === 1 ? 'Tomorrow' : 
                           pred.day === 2 ? 'Day After' :
                           `Day ${pred.day}`;
            
            card.innerHTML = `
                <div class="day">${dayText}</div>
                <div class="price">₹${pred.price}</div>
                <div class="change ${pred.percent_change >= 0 ? 'positive' : 'negative'}">
                    ${pred.percent_change >= 0 ? '↑' : '↓'} ${Math.abs(pred.percent_change)}%
                </div>
            `;
            
            predictionsGrid.appendChild(card);
        });

        // Show result section
        result.classList.remove('hidden');
    } catch (err) {
        showError('Failed to get prediction. Please try again.');
    } finally {
        loading.classList.add('hidden');
    }
}

function showError(message) {
    const error = document.getElementById('error');
    error.querySelector('p').textContent = message;
    error.classList.remove('hidden');
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('result').classList.add('hidden');
}
