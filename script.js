let percentage = 0;
const percentageElement = document.getElementById('percentage');
const loadingContainer = document.getElementById('loading-container');
const landingContainer = document.getElementById('landing-container');

const loadingInterval = setInterval(() => {
    if (percentage < 100) {
        percentage++;
        percentageElement.textContent = percentage + '%';
    } else {
        clearInterval(loadingInterval);
        loadingContainer.style.display = 'none';
        landingContainer.style.display = 'block';
    }
}, 100); // Adjust the interval for speed of loading

function toggleForms() {
    const signInForm = document.getElementById('sign-in-form');
    const signUpForm = document.getElementById('sign-up-form');

    if (signInForm.style.display === 'none') {
        signInForm.style.display = 'block';
        signUpForm.style.display = 'none';
    } else {
        signInForm.style.display = 'none';
        signUpForm.style.display = 'block';
    }
}
