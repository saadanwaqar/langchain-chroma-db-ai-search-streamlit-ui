# PDF Question Answering System

This repository contains two versions of a PDF Question Answering system built with Streamlit and LangChain:
1. **ChromaDB Version** - Uses local vector storage.
2. **Azure AI Search Version** - Uses cloud-based vector storage.

Both systems allow users to upload PDFs, process them, and ask questions about their content using natural language.

## Common Requirements (Both Versions)

### Prerequisites
- Python 3.8 or higher
- Azure OpenAI API access

### Base Packages
```bash
pip install streamlit python-dotenv PyPDF2 langchain langchain-openai
```

### Common Environment Variables
Create a `.env` file in your project root with these variables:
```env
AZURE_OPENAI_API_VERSION=your-api-version
AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=your-endpoint
```

## Version 1: ChromaDB Implementation

### Additional Installation
```bash
pip install chromadb
```

### Features
- Local vector storage using ChromaDB
- No additional cloud services required
- Persistent storage in the local directory
- Easy to set up and run locally

### Directory Structure
```
project/
├── .env
├── app.py
└── chroma_db/  # Created automatically when running
```

### Running the Application
```bash
streamlit run app.py
```

### Cleanup
- The `chroma_db` directory can be manually deleted to clear the vector store.
- Alternatively, use the "Clear PDF and Start Over" button in the application.

## Version 2: Azure AI Search Implementation

### Additional Installation
```bash
pip install azure-identity azure-search-documents azure-core
```

### Additional Environment Variables
Add these to your `.env` file:
```env
AZURE_AI_SEARCH_SERVICE=your-search-service-name
AZURE_AI_SEARCH_KEY=your-search-admin-key
```

### Features
- Cloud-based vector storage using Azure AI Search
- Scalable and managed by Azure
- Persistent across sessions
- Suitable for production deployments

### Prerequisites
1. Azure Account
2. Azure AI Search service instance
3. Appropriate permissions to create/delete indexes

### Directory Structure
```
project/
├── .env
└── app.py
```

### Running the Application
```bash
streamlit run app.py
```

### Cleanup
- Use the "Clear Index and Start Over" button in the application.
- Alternatively, delete the index manually from the Azure AI Search portal.

## Usage (Both Versions)

1. Start the application.
2. Upload a PDF file.
3. Wait for processing to complete.
4. Enter questions about the PDF content.
5. View answers and source context.

## Key Differences

### ChromaDB Version
- **Pros:**
  - Simpler setup
  - No cloud costs
  - Works offline
- **Cons:**
  - Limited by local storage
  - Not suitable for large-scale deployments
  - No built-in redundancy

### Azure AI Search Version
- **Pros:**
  - Highly scalable
  - Cloud-based reliability
  - Better for production use
  - Built-in redundancy
- **Cons:**
  - Requires Azure subscription
  - More complex setup
  - Associated cloud costs

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**
   - Ensure the PDF is readable and not encrypted.
   - Ensure sufficient memory for large PDFs.

2. **Authentication Errors**
   - Verify environment variables are set correctly.
   - Check Azure credentials and permissions.

3. **Vector Store Issues**
   - ChromaDB: Check write permissions in the local directory.
   - Azure: Verify the search service is running and accessible.

### Azure AI Search Specific
- Ensure the service name and key are correct.
- Check if you have sufficient permissions to create/delete indexes.
- Verify network connectivity to Azure services.

### ChromaDB Specific
- Ensure sufficient disk space.
- Check folder permissions.
- Verify ChromaDB installation is complete.

## License

This project is licensed under the MIT License.
