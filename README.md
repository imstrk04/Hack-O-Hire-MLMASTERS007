# Password Strength Analyzer

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24.0-red)

## 🔒 Advanced Password Security Analysis Tool

A comprehensive, user-friendly password strength analysis tool that provides real-time feedback, security metrics, and actionable recommendations to create stronger passwords.

### Developed by ML MASTERS 007
- Sadakopa Ramakrishnan
- Shivani Garimella
- Santhosh Vasudevan

## 🌟 Features

### Password Strength Analysis
- **Entropy Calculation**: Evaluates password strength based on character diversity, length, and pattern complexity
- **Visual Strength Meter**: Easy-to-understand categorization from Very Weak to Very Strong
- **Pattern Recognition**: Identifies common vulnerabilities like keyboard patterns, repeated sequences, and dictionary words

### Time-to-Crack Estimation
- **Multiple Attack Scenarios**: Estimates based on different attack vectors (online throttled, offline fast hash, etc.)
- **Advanced Algorithm Consideration**: Adjusts estimates based on bcrypt, SHA-256, and other modern hashing algorithms
- **Visual Timeline**: Graphical representation of time-to-crack estimates

### Data Breach Check
- **Compromised Password Detection**: Simulates checking against leaked password databases
- **Clear Risk Indicators**: Immediate warnings for passwords found in known data breaches

### Real-Time Feedback
- **Instant Analysis**: Character-by-character evaluation with immediate feedback
- **Actionable Suggestions**: Specific recommendations to improve password strength
- **Visual Indicators**: Color-coded feedback for intuitive understanding

### AI-Powered Password Aging Reminder
- **Smart Expiration Calculation**: Recommends password change timeline based on strength and security context
- **Trend-Based Analysis**: Considers evolving attack methodologies to recommend optimal password rotation

### Custom Password Generation
- **Strength-Based Suggestions**: Generates strong alternatives that meet specific entropy requirements
- **Personalized Options**: Create passwords that balance security and memorability

### User-Friendly Interface
- **Interactive Dashboard**: Clean, intuitive layout with expandable sections
- **Data Visualizations**: Charts for entropy, character distribution, and strength metrics
- **Responsive Design**: Works across devices and screen sizes

## 📊 Technical Features

- **Entropy Analysis**: Shannon entropy calculation with character set evaluation
- **Pattern Detection**: Regex-based identification of common password weaknesses
- **Visualization Tools**: Real-time charts showing password composition and security metrics
- **Hash Simulation**: Estimation of cracking difficulty across multiple hashing algorithms
- **Custom Security Models**: ML-based evaluation of password strength beyond simple rules

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/ml-masters-007/password-strength-analyzer.git
cd password-strength-analyzer
```

2. Create and activate virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
streamlit run app.py
```

5. Open your browser and navigate to http://localhost:8501

## 📋 Usage

1. Enter a password in the input field
2. View real-time analysis of password strength, including:
   - Overall strength score
   - Entropy calculation
   - Character composition
   - Vulnerability assessment
   - Time-to-crack estimates
   - Improvement suggestions
3. Generate stronger password alternatives
4. Receive guidance on when to change your password

## 🔧 Technical Implementation

### Core Components
- **Streamlit**: Web application framework
- **NumPy/Pandas**: Data processing and analysis
- **Plotly/Matplotlib**: Data visualization
- **NLTK/RegEx**: Pattern recognition and dictionary attack simulation
- **hashlib/bcrypt**: Password hashing and security simulation

### Security Considerations
- Passwords are processed locally in memory
- No passwords are stored or sent to external servers
- Simulated breach check (for production, integrate with a secure API like HaveIBeenPwned)

## 🛠️ Future Development

- API integration with HaveIBeenPwned for actual breach checking
- Multi-factor authentication recommendation engine
- Enterprise version with organization-wide password policy enforcement
- Mobile application with fingerprint/face ID integration
- Browser extension for real-time password evaluation during account creation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- Shannon, C. E. (1948). A Mathematical Theory of Communication
- NIST Special Publication 800-63B: Digital Identity Guidelines
- Bonneau, J. (2012). The Science of Guessing: Analyzing an Anonymized Corpus of 70 Million Passwords
- Wheeler, D. L. (2016). zxcvbn: Low-Budget Password Strength Estimation

## 📞 Contact

For any questions or support, please reach out to:
- sadakopa2210221@ssn.edu.in
