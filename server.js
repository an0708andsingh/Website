const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors());

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/signup-demo', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
});

// User Schema
const userSchema = new mongoose.Schema({
    username: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
});

const User = mongoose.model('User', userSchema);

// API Endpoint for Sign-Up
app.post('/api/signup', async (req, res) => {
    try {
        const { username, email, password } = req.body;

        // Check if email already exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.json({ success: false, message: 'Email already registered' });
        }

        // Hash the password
        const hashedPassword = await bcrypt.hash(password, 10);

        // Save user to database
        const newUser = new User({ username, email, password: hashedPassword });
        await newUser.save();

        res.json({ success: true, message: 'User registered successfully!' });
    } catch (error) {
        res.json({ success: false, message: error.message });
    }
});

// Start Server
app.listen(3000, () => {
    console.log('Server running on http://localhost:3000');
});
