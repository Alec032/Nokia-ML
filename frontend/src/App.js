import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import './App.css';

export default function App() {
  const [formData, setFormData] = useState({
    choice: '',
    title: '',
    description: '',
    build: '',
    feature: '',
    release: '',
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    
    if (type === 'file') {
      const file = e.target.files[0];
      setFormData({ ...formData, json_file_path: file.name });
    } else {
      setFormData({ ...formData, [name]: value });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://127.0.0.1:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      if (response.ok) {
        setPredictionResult(data); // Update with data directly, not data[0]
        setError(null);
      } else {
        setError(data.error || 'Prediction failed'); // Assuming the error message is in the 'error' field
      }
    } catch (error) {
      console.error('Error:', error);
      setError('Prediction failed');
    }
  };

  return (
    <div className="page">
      <img src="https://www.nokia.com/themes/custom/onenokia_reskin/logo.svg" alt="Home" width="140" height="33" className="logo" />
      <div className='wrapper'>
        <div className="container">
          {predictionResult && (
            <div className="prediction-box">
              <div className="prediction">
                <p>Group in charge: {predictionResult.label}</p>
                <p>Nivel de incredere: {predictionResult.probability}%</p>
              </div>
            </div>
          )}
          <form onSubmit={handleSubmit}>
            <div className="field" tabIndex="1">
              <label htmlFor="choice">Select an option:</label>
              <select name="choice" value={formData.choice} onChange={handleChange}>
                <option value="1">Send Pronto</option>
                <option value="2">Send Json</option>
              </select>
            </div>
            {formData.choice === '1' && (
              <>
                <div className="field" tabIndex="2">
                  <label htmlFor="title">Title</label>
                  <input name="title" type="text" required onChange={handleChange} />
                </div>
                <div className="field" tabIndex="3">
                  <label htmlFor="build">Build</label>
                  <input name="build" type="text" required onChange={handleChange} />
                </div>
                <div className="field" tabIndex="4">
                  <label htmlFor="description">Description</label>
                  <textarea name="description" required onChange={handleChange}></textarea>
                </div>
                <div className="field" tabIndex="5">
                  <label htmlFor="release">Release</label>
                  <input name="release" type="text" required onChange={handleChange} />
                </div>
                <div className="field" tabIndex="6">
                  <label htmlFor="feature">Feature</label>
                  <input name="feature" type="text" required onChange={handleChange} />
                </div>
              </>
            )}
            {formData.choice === '2' && (
              <div className="field" tabIndex="6">
                <label htmlFor="jsonFilePath">Upload JSON File</label>
                <input name="jsonFilePath" type="file" accept="application/json" onChange={handleChange} />
              </div>
            )}
            <button type="submit">Predict</button>
          </form>
          {error && <p className="error">{error}</p>}
        </div>
      </div>
    </div>
  );  
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
