# Complete ML Analysis System

A comprehensive machine learning analysis system that performs clustering, classification, comparison, and interactive analysis of data quality issues using AWS Bedrock and advanced ML techniques.

## 🎯 Overview

This system provides end-to-end analysis of data quality issues with the following capabilities:

- **Clustering Analysis**: Uses Amazon Titan embeddings and UMAP+HDBSCAN for intelligent clustering
- **MLflow Integration**: Saves and reuses optimized parameters for consistent results
- **Zero-shot Classification**: Automatically assigns thematic labels to clusters using Llama 3
- **Binary Classification**: Identifies data quality issues with Yes/No classification
- **Cluster Comparison**: Finds similar and unique clusters between datasets
- **RAG Chat Interface**: Interactive chat with full conversation history and data context
- **Excel Reporting**: Comprehensive reports with multiple tabs and detailed analysis
- **Vector Database Caching**: ChromaDB integration for efficient embedding storage

## 🚀 Quick Start

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **AWS Configuration**:
   - Configure AWS credentials: `aws configure`
   - Or set environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

3. **Data Files**:
   - Place `data_issues.csv` and `issue_summaries.csv` in the project directory

4. **Configuration**:
   - Ensure `config/config.yaml` exists with AWS settings

### Basic Usage

```bash
# Run complete analysis with default settings
python complete_ml_analysis.py

# Run with custom similarity threshold
python complete_ml_analysis.py 0.4

# Start interactive RAG chat
python complete_ml_analysis.py --rag-chat

# View MLflow UI
python complete_ml_analysis.py --mlflow-ui
```

## 📊 Features

### 1. Clustering Analysis
- **Embedding Generation**: Uses Amazon Titan embeddings (1024 dimensions)
- **Dimensionality Reduction**: UMAP for efficient clustering
- **Density-based Clustering**: HDBSCAN for robust cluster detection
- **Parameter Optimization**: Optuna for automatic hyperparameter tuning
- **Caching**: ChromaDB for embedding storage and retrieval

### 2. MLflow Integration
- **Parameter Persistence**: Saves optimized clustering parameters
- **Consistent Results**: Reuses saved parameters to prevent re-training
- **Experiment Tracking**: Tracks all runs, parameters, and metrics
- **Web UI**: Visual interface for experiment management
- **Model Registry**: Version control for clustering models

### 3. Zero-shot Classification
- **Automatic Theming**: Uses Llama 3 to generate 1-3 word cluster themes
- **All-text Context**: Analyzes all texts in each cluster
- **No Training Required**: Works without labeled examples
- **Fast Processing**: Optimized for large datasets (50k+ records)

### 4. Binary Classification
- **Data Quality Detection**: Identifies records as data quality issues (Yes/No)
- **Individual Analysis**: Processes each record separately
- **Parallel Processing**: Uses ThreadPoolExecutor for speed
- **Caching**: Stores classification results for reuse

### 5. Cluster Comparison
- **Similarity Analysis**: Finds similar clusters between datasets
- **Unique Detection**: Identifies unique clusters in each dataset
- **Overlap Calculation**: Measures cluster overlap percentage
- **Detailed Reporting**: Comprehensive comparison summaries

### 6. RAG Chat Interface
- **Interactive Chat**: Natural language queries about your data
- **Context Retrieval**: Semantic search through all analyzed data
- **Conversation History**: Maintains chat context and recollection
- **Streaming Responses**: Real-time LLM responses
- **Data Integration**: Leverages clusters, themes, and classifications

### 7. Excel Reporting
- **Multiple Tabs**: Comprehensive analysis across different views
- **Original Data**: Preserves source data with cluster and theme columns
- **Comparison Analysis**: Similar and unique clusters with full details
- **Summary Statistics**: Statistical overview of the analysis
- **Temporal Analysis**: Date-based insights (if available)

## 🎛️ Command Line Options

### Basic Analysis
```bash
# Default analysis
python complete_ml_analysis.py

# Custom similarity threshold
python complete_ml_analysis.py 0.4

# Clear cache before analysis
python complete_ml_analysis.py --clear-cache

# Rebuild cache
python complete_ml_analysis.py --rebuild-cache 0.5
```

### MLflow Management
```bash
# Start MLflow UI
python complete_ml_analysis.py --mlflow-ui

# Custom port for MLflow UI
python complete_ml_analysis.py --mlflow-ui 8080
```

### Database Management
```bash
# Reset ChromaDB
python complete_ml_analysis.py --reset-db

# Force reset with cleanup
python complete_ml_analysis.py --force-reset
```

### Classification Options
```bash
# Run only classification (requires existing clustering)
python complete_ml_analysis.py --optimized-classify

# Custom number of workers
python complete_ml_analysis.py --optimized-classify 8

# Custom workers and similarity threshold
python complete_ml_analysis.py --optimized-classify 8 0.3
```

### Interactive Features
```bash
# Start RAG chat
python complete_ml_analysis.py --rag-chat

# Test cache functionality
python complete_ml_analysis.py --test-cache
```

### Help and Information
```bash
# Show all options
python complete_ml_analysis.py --help
```

## 📁 Output Structure

### Generated Files
```
complete_analysis_results/
├── clustering_results.json          # Clustering evaluation and parameters
├── comparison_results.json          # Detailed cluster comparison
├── comparison_summary.csv           # Summary table of comparisons
├── cluster_classifications.json     # Zero-shot classification results
├── cluster_themes.csv              # Theme assignments
├── binary_classifications.json     # Yes/No classifications
├── clustering_analysis_report.xlsx # Comprehensive Excel report
└── *.npy files                     # Numpy arrays (embeddings, labels)

chroma_db/                          # Vector database cache
├── chroma.sqlite3                  # SQLite database
└── embeddings/                     # Cached embeddings

mlruns/                             # MLflow experiment tracking
└── clustering_analysis/            # Experiment runs and models
```

### Excel Report Tabs
1. **Original Data Issues**: Source data with cluster and theme columns
2. **Original Issue Summaries**: Source data with cluster and theme columns
3. **Similar Clusters Analysis**: Data from both files with similar clusters
4. **Unique Clusters Analysis**: Data from both files with unique clusters
5. **Summary Statistics**: Statistical overview of the analysis
6. **Temporal Analysis**: Date-based insights (if available)
7. **Cluster Details**: Detailed cluster information and metrics

## 🔧 Configuration

### AWS Configuration (`config/config.yaml`)
```yaml
aws:
  region: us-east-1
  bedrock:
    embedding_model_id: amazon.titan-embed-text-v2:0
    model_id: meta.llama3-8b-instruct-v1:0
    max_tokens: 4096
    temperature: 0.1
```

### MLflow Configuration
- **Tracking URI**: `sqlite:///mlflow.db`
- **Experiment Name**: `clustering_analysis`
- **Model Registry**: `clustering_models`

### ChromaDB Configuration
- **Database Path**: `./chroma_db`
- **Embedding Dimension**: 1024 (Amazon Titan v2)
- **Collection Names**: `data_issues`, `issue_summaries`

## 📈 Performance

### Processing Times
- **First Run**: 2-5 minutes (generates embeddings)
- **Subsequent Runs**: 30-60 seconds (uses cached embeddings)
- **Classification**: 1-2 minutes for 100 records
- **RAG Chat**: Real-time responses

### Resource Usage
- **Memory**: ~500MB for 100 records per dataset
- **Storage**: ~50MB for embeddings and cache
- **API Calls**: Optimized to minimize AWS costs

## 🐛 Troubleshooting

### Common Issues

1. **AWS Credentials Not Found**
   ```bash
   aws configure
   # Or set environment variables
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```

2. **ChromaDB File Lock Issues**
   ```bash
   python complete_ml_analysis.py --reset-db
   # Or force reset
   python complete_ml_analysis.py --force-reset
   ```

3. **MLflow UI Not Starting**
   ```bash
   # Check if MLflow is installed
   pip install mlflow
   
   # Try different port
   python complete_ml_analysis.py --mlflow-ui 8080
   ```

4. **Empty Embeddings in ChromaDB**
   ```bash
   # Rebuild cache
   python complete_ml_analysis.py --rebuild-cache
   ```

5. **RAG Chat Not Working**
   ```bash
   # Ensure clustering is completed first
   python complete_ml_analysis.py
   # Then start RAG chat
   python complete_ml_analysis.py --rag-chat
   ```

### Debug Mode
Enable debug output by modifying the logging level in the script or adding verbose flags.

## 🎯 Use Cases

### Data Quality Analysis
- Identify patterns in data quality issues
- Categorize issues by theme and severity
- Track issue trends over time
- Generate actionable insights

### Cluster Analysis
- Find similar problems across datasets
- Identify unique issues in specific areas
- Measure overlap between different data sources
- Optimize clustering parameters

### Interactive Exploration
- Ask natural language questions about your data
- Explore clusters and themes interactively
- Get insights on specific records or patterns
- Maintain conversation context

### Reporting and Documentation
- Generate comprehensive Excel reports
- Create visual summaries of analysis
- Document findings for stakeholders
- Track analysis history and changes

## 🔮 Future Enhancements

### Planned Features
- **Advanced Visualization**: Interactive dashboards and charts
- **Multi-language Support**: Analysis in different languages
- **Real-time Processing**: Stream processing for live data
- **Advanced Analytics**: Statistical significance testing
- **API Integration**: REST API for programmatic access

### Integration Possibilities
- **Kubernetes**: Containerized deployment
- **AWS SageMaker**: Managed ML endpoints
- **Docker**: Containerization for easy deployment
- **CI/CD**: Automated analysis pipelines

## 📚 Additional Resources

### Documentation
- [README_COMPLETE_ANALYSIS.md](README_COMPLETE_ANALYSIS.md) - Detailed clustering analysis
- [README_MLFLOW_INTEGRATION.md](README_MLFLOW_INTEGRATION.md) - MLflow integration guide
- [README_ZERO_SHOT_CLASSIFICATION.md](README_ZERO_SHOT_CLASSIFICATION.md) - Classification features

### External Resources
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the detailed README files
3. Check the MLflow UI for experiment details
4. Use the `--help` option for command-line assistance

---

**Happy Analyzing! 🚀**
