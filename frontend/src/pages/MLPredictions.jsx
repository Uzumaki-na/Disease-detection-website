import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import MLPrediction from '../components/MLPrediction';

const MLPredictions = () => {
  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Medical Image Analysis
        </Typography>
        
        <Typography variant="body1" paragraph align="center">
          Upload medical images for AI-powered analysis. Our models can help detect skin cancer
          and malaria from appropriate medical images.
        </Typography>

        <MLPrediction 
          title="Skin Cancer Detection" 
          endpoint="/predict_skin_cancer" 
        />
        
        <MLPrediction 
          title="Malaria Detection" 
          endpoint="/predict_malaria" 
        />
      </Box>
    </Container>
  );
};

export default MLPredictions;
