function submitSignUp() {
    const username = document.getElementById('username').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const responseMessage = document.getElementById('response-message');

    fetch('http://localhost:3000/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                responseMessage.textContent = 'Sign-Up Successful!';
            } else {
                responseMessage.textContent = 'Error: ' + data.message;
            }
        })
        .catch(error => {
            responseMessage.textContent = 'An error occurred. Please try again.';
            console.error('Error:', error);
        });
}
