import React from 'react';
import { Container, Typography, Box, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Home = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuth();

  return (
    <Container maxWidth="md">
      <Box
        sx={{
          mt: 8,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        <Typography variant="h3" component="h1" gutterBottom>
          Health Risk Assessment
        </Typography>
        <Typography variant="h5" component="h2" gutterBottom color="text.secondary">
          AI-Powered Medical Image Analysis
        </Typography>
        <Typography variant="body1" paragraph sx={{ mt: 2 }}>
          Our advanced machine learning models can help detect potential health risks
          through medical image analysis, including skin cancer detection and malaria cell identification.
        </Typography>
        
        {!isAuthenticated ? (
          <Box sx={{ mt: 4, display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              size="large"
              onClick={() => navigate('/register')}
            >
              Get Started
            </Button>
            <Button
              variant="outlined"
              size="large"
              onClick={() => navigate('/login')}
            >
              Login
            </Button>
          </Box>
        ) : (
          <Button
            variant="contained"
            size="large"
            onClick={() => navigate('/ml-predictions')}
            sx={{ mt: 4 }}
          >
            Go to Analysis
          </Button>
        )}
      </Box>
    </Container>
  );
};

export default Home;
