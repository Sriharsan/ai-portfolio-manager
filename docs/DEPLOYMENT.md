# Deployment Guide

## Local Development

### Prerequisites
- Python 3.8+
- Git
- 4GB+ RAM recommended

### Setup Steps
1. **Clone Repository**
   ```bash
   git clone https://github.com/Sriharsan/ai-portfolio-manager.git
   cd ai-portfolio-manager
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

5. **Run Application**
   ```bash
   streamlit run streamlit_app.py
   ```

## Streamlit Cloud Deployment

### Step 1: Prepare Repository
- Ensure all files are committed to GitHub
- Verify requirements.txt is complete
- Add secrets to Streamlit Cloud dashboard

### Step 2: Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub repository
3. Select `streamlit_app.py` as main file
4. Add secrets in "Secrets" section:
   ```toml
   ALPHA_VANTAGE_API_KEY = "your_key"
   FRED_API_KEY = "your_key"
   HUGGINGFACE_API_KEY = "your_key"
   ```

### Step 3: Configure
- Set Python version to 3.8+
- Verify all dependencies install correctly
- Test application functionality

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run
```bash
docker build -t ai-portfolio-manager .
docker run -p 8501:8501 --env-file .env ai-portfolio-manager
```

## Cloud Platform Deployment

### AWS EC2
1. Launch EC2 instance (t3.medium recommended)
2. Install Python 3.8+ and Git
3. Clone repository and setup as above
4. Use screen/tmux for persistent session
5. Configure security group for port 8501

### Google Cloud Run
1. Create `cloudbuild.yaml`:
   ```yaml
   steps:
   - name: 'gcr.io/cloud-builders/docker'
     args: ['build', '-t', 'gcr.io/$PROJECT_ID/ai-portfolio-manager', '.']
   - name: 'gcr.io/cloud-builders/docker'
     args: ['push', 'gcr.io/$PROJECT_ID/ai-portfolio-manager']
   images:
   - 'gcr.io/$PROJECT_ID/ai-portfolio-manager'
   ```

2. Deploy:
   ```bash
   gcloud run deploy --image gcr.io/PROJECT_ID/ai-portfolio-manager --platform managed
   ```

### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

## Performance Optimization

### Memory Management
- Limit pandas DataFrame sizes
- Use data caching strategically
- Monitor memory usage with profiling tools

### API Rate Limits
- Implement exponential backoff
- Cache API responses
- Use multiple API keys for redundancy

### Monitoring
- Set up logging to files
- Monitor application health
- Track API usage and costs

## Security Considerations

### API Keys
- Never commit API keys to repository
- Use environment variables or secrets management
- Rotate keys regularly

### Data Protection
- Encrypt sensitive data at rest
- Use HTTPS for all communications
- Implement proper authentication for production

## Troubleshooting

### Common Issues
1. **Import Errors**: Verify Python path and dependencies
2. **API Failures**: Check API keys and rate limits
3. **Memory Issues**: Reduce data size or optimize code
4. **Slow Performance**: Enable caching and optimize queries

### Logs Location
- Local: Console output
- Streamlit Cloud: Application logs in dashboard
- Docker: Use `docker logs <container_id>`

### Health Checks
The application includes health check endpoints for monitoring deployment status.