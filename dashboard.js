// All dashboard interactivity and dynamic rendering

document.addEventListener('DOMContentLoaded', function () {
    // Example data (replace with API calls or dynamic sources in production)
    const summaryData = [
        { icon: 'fa-boxes', color: 'text-primary', title: 'Total Items', value: 2547, change: '+3.2%', changeIcon: 'fa-arrow-up', changeClass: 'text-success', subtitle: 'from last month' },
        { icon: 'fa-check-circle', color: 'text-success', title: 'Items in Stock', value: 1893, subtitle: '74.3% of total inventory' },
        { icon: 'fa-exclamation-triangle', color: 'text-warning', title: 'Low Stock Items', value: 342, subtitle: '13.4% of total inventory' },
        { icon: 'fa-times-circle', color: 'text-danger', title: 'Out of Stock', value: 312, subtitle: '12.3% of total inventory' }
    ];

    // TODO: Render summary cards dynamically
    // TODO: Render inventory table dynamically
    // TODO: Render recent activity and expiring items dynamically
    // TODO: Add search/filter interactivity
    // TODO: Make charts update live based on data

    // Category Chart
    const categoryCtx = document.getElementById('categoryChart').getContext('2d');
    const categoryChart = new Chart(categoryCtx, {
        type: 'doughnut',
        data: {
            labels: ['PPE', 'Medication', 'Supplies', 'Equipment', 'Fluids'],
            datasets: [{
                data: [1042, 328, 714, 215, 980],
                backgroundColor: [
                    '#3498db',
                    '#2ecc71',
                    '#e74c3c',
                    '#f39c12',
                    '#8e44ad'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Trend Chart
    const trendCtx = document.getElementById('trendChart').getContext('2d');
    const trendChart = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [
                {
                    label: 'PPE',
                    data: [1200, 1350, 1500, 1650, 1800, 1750],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Medication',
                    data: [800, 850, 900, 950, 1000, 1050],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Supplies',
                    data: [600, 650, 700, 750, 800, 850],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Example: Add more dynamic features here (search, filter, live updates)
});
