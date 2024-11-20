import React from 'react';
import { Box, Card, CardContent, Typography, Container, Grid, Paper } from '@mui/material';
import { useAuth } from '../contexts/AuthContext';

const Profile = () => {
  const { user } = useAuth();

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        Profile
      </Typography>
      
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" color="textSecondary">
                First Name
              </Typography>
              <Typography variant="body1" sx={{ mb: 2 }}>
                {user?.firstName || 'N/A'}
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" color="textSecondary">
                Last Name
              </Typography>
              <Typography variant="body1" sx={{ mb: 2 }}>
                {user?.lastName || 'N/A'}
              </Typography>
            </Grid>
            
            <Grid item xs={12}>
              <Typography variant="subtitle1" color="textSecondary">
                Email
              </Typography>
              <Typography variant="body1">
                {user?.email || 'N/A'}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      <Typography variant="h5" gutterBottom sx={{ mt: 4 }}>
        Recent Activity
      </Typography>
      
      <Paper sx={{ p: 2 }}>
        <Typography variant="body1" color="textSecondary">
          No recent activity to display.
        </Typography>
      </Paper>
    </Container>
  );
};

export default Profile;
