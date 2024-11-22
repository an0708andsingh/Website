let percentage = 0;
const percentageElement = document.getElementById('percentage');

const loadingInterval = setInterval(() => {
    if (percentage < 100) {
        percentage++;
        percentageElement.textContent = percentage + '%';
    } else {
        clearInterval(loadingInterval);
        // You can redirect or perform another action here after loading is complete
    }
}, 100); // Adjust the interval for speed of loading
