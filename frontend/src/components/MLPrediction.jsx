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
      const response = await fetch(`${import.meta.env.VITE_API_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get prediction');
      }
      
      const result = await response.json();
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
    const result = prediction.prediction > 0.5 ? 
      (title.includes('Skin Cancer') ? 'Malignant' : 'Parasitized') :
      (title.includes('Skin Cancer') ? 'Benign' : 'Uninfected');

    return (
      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Typography variant="h6" color={result === 'Malignant' || result === 'Parasitized' ? 'error' : 'success'}>
          Result: {result}
        </Typography>
        <Typography variant="body1">
          Confidence: {confidence}%
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Prediction made at: {new Date(prediction.timestamp).toLocaleString()}
        </Typography>
      </Box>
    );
  };

  return (
    <Card sx={{ maxWidth: 600, mx: 'auto', my: 2 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          {title}
        </Typography>
        
        <Box sx={{ my: 2 }}>
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
              color={error ? 'error' : 'primary'}
            >
              Upload Image
            </Button>
          </label>
        </Box>

        {error && (
          <Typography color="error" variant="body2" sx={{ mt: 1 }}>
            {error}
            {retryCount < maxRetries && (
              <Button
                size="small"
                onClick={handleRetry}
                sx={{ ml: 1 }}
              >
                Retry
              </Button>
            )}
          </Typography>
        )}

        {preview && (
          <Box sx={{ textAlign: 'center' }}>
            <ImagePreview src={preview} alt="Preview" />
          </Box>
        )}

        {selectedFile && (
          <Box sx={{ mt: 2, textAlign: 'center' }}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <>
                  <CircularProgress size={24} sx={{ mr: 1 }} />
                  Processing...
                </>
              ) : (
                'Get Prediction'
              )}
            </Button>
          </Box>
        )}

        {renderPredictionResult()}
      </CardContent>
    </Card>
  );
};

export default MLPrediction;
