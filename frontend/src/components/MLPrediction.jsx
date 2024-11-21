import React, { useState } from 'react';
import { Box, Button, Card, CardContent, Typography, CircularProgress } from '@mui/material';
import { styled } from '@mui/system';

const Input = styled('input')({
  display: 'none',
});

const ImagePreview = styled('img')({
  maxWidth: '100%',
  maxHeight: '300px',
  marginTop: '20px',
});

const MLPrediction = ({ title, endpoint }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);
  const maxRetries = 3;
  const maxSizeMB = 10;

  const validateFile = (file) => {
    // Check file type
    if (!file.type.startsWith('image/')) {
      throw new Error('Please select an image file (PNG, JPG, etc.)');
    }

    // Check file size
    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > maxSizeMB) {
      throw new Error(`File size (${sizeMB.toFixed(1)}MB) exceeds maximum allowed size (${maxSizeMB}MB)`);
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      validateFile(file);
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    } catch (error) {
      setError(error.message);
      setSelectedFile(null);
      setPreview('');
    }
  };

  const handleRetry = () => {
    if (retryCount < maxRetries) {
      setRetryCount(prev => prev + 1);
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      console.log('Making request to:', `${import.meta.env.VITE_API_URL}${endpoint}`);
      const response = await fetch(`${import.meta.env.VITE_API_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }
      
      const result = await response.json();
      console.log('Prediction result:', result);
      setPrediction(result);
      setRetryCount(0);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Failed to get prediction');
      
      // Auto-retry on network errors
      if (retryCount < maxRetries && error.message.includes('network')) {
        setTimeout(handleRetry, 1000 * (retryCount + 1));
      }
    } finally {
      setLoading(false);
    }
  };

  const renderPredictionResult = () => {
    if (!prediction) return null;

    const confidence = (prediction.confidence * 100).toFixed(1);
    const result = prediction.prediction > 0.5 ? 'Positive' : 'Negative';
    const timestamp = new Date(prediction.timestamp).toLocaleString();

    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Prediction Result:
        </Typography>
        <Typography>
          Result: {result}
        </Typography>
        <Typography>
          Confidence: {confidence}%
        </Typography>
        <Typography variant="caption" display="block">
          Timestamp: {timestamp}
        </Typography>
      </Box>
    );
  };

  return (
    <Card sx={{ mb: 4 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          {title}
        </Typography>

        <Box sx={{ mt: 2 }}>
          <label htmlFor={`upload-${title}`}>
            <Input
              accept="image/*"
              id={`upload-${title}`}
              type="file"
              onChange={handleFileSelect}
              disabled={loading}
            />
            <Button
              variant="contained"
              component="span"
              disabled={loading}
            >
              Select Image
            </Button>
          </label>

          {preview && (
            <Box sx={{ mt: 2 }}>
              <ImagePreview src={preview} alt="Preview" />
            </Box>
          )}

          {selectedFile && (
            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={loading}
              sx={{ mt: 2, ml: 2 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Analyze'}
            </Button>
          )}

          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              Error: {error}
            </Typography>
          )}

          {renderPredictionResult()}
        </Box>
      </CardContent>
    </Card>
  );
};

export default MLPrediction;
