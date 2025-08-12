# Complete ML Analysis: Clustering + Cluster Comparison
# This script performs clustering analysis using Amazon Titan embeddings via AWS Bedrock
# and then compares clusters between datasets to identify similarities and unique clusters

import os
import json
import yaml
import warnings
import hashlib
import re
import sys
import subprocess
import threading
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import openpyxl


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    PURPLE = "\033[35m"
    ORANGE = "\033[33m"
    PINK = "\033[95m"
    LIGHT_BLUE = "\033[94m"
    LIGHT_GREEN = "\033[92m"
    LIGHT_RED = "\033[91m"
    LIGHT_YELLOW = "\033[93m"


# ML and Clustering imports
from sklearn.metrics import (
    silhouette_score,
    pairwise_distances,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
import optuna
import mlflow
import mlflow.sklearn

# Set style for better plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

print(
    f"{Colors.HEADER}{Colors.BOLD}🚀 Complete ML Analysis: Clustering + Cluster Comparison{Colors.ENDC}"
)
print(f"{Colors.OKBLUE}{'=' * 60}{Colors.ENDC}")


class CompleteMLAnalysis:
    """
    Complete ML Analysis class that handles both clustering and cluster comparison.
    """

    def __init__(self):
        self.data_issues = None
        self.issue_summaries = None
        self.config = None
        self.bedrock_client = None
        self.embedding_model_id = None
        self.chroma_client = None
        self.data_issues_collection = None
        self.issue_summaries_collection = None

        # MLflow configuration
        self.mlflow_tracking_uri = "sqlite:///mlflow.db"
        self.experiment_name = "clustering_analysis"
        self.model_registry_name = "clustering_models"

        # Initialize MLflow
        self.setup_mlflow()

        # Clustering results
        self.issue_embeddings = None
        self.llama_model_id = None
        self.summary_embeddings = None
        self.issue_umap = None
        self.issue_clusters = None
        self.summary_umap = None
        self.summary_clusters = None
        self.issue_evaluation = None
        self.summary_evaluation = None
        self.issue_best_params = None
        self.summary_best_params = None

        # Comparison results
        self.comparison_results = {}

        # Preprocessing results
        self.issue_indices = None
        self.summary_indices = None

    def setup_mlflow(self):
        """Initialize MLflow tracking and experiment setup."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

            # Create or get experiment
            mlflow.set_experiment(self.experiment_name)

            print(f"{Colors.OKGREEN}✅ MLflow initialized successfully{Colors.ENDC}")
            print(
                f"{Colors.OKBLUE}   Tracking URI: {self.mlflow_tracking_uri}{Colors.ENDC}"
            )
            print(f"{Colors.OKBLUE}   Experiment: {self.experiment_name}{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.WARNING}⚠️  MLflow setup failed: {e}{Colors.ENDC}")
            print(f"{Colors.WARNING}   Continuing without MLflow tracking{Colors.ENDC}")

    def check_existing_model(self, dataset_name):
        """Check if an optimized model exists in MLflow for the given dataset."""
        try:
            # Search for existing runs with the dataset name
            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                filter_string=f"tags.dataset_name = '{dataset_name}' AND tags.status = 'completed'",
                order_by=["metrics.silhouette_score DESC"],
            )

            if not runs.empty:
                best_run = runs.iloc[0]
                run_id = best_run["run_id"]

                print(
                    f"{Colors.OKGREEN}✅ Found existing optimized model for {dataset_name}{Colors.ENDC}"
                )
                print(f"{Colors.OKBLUE}   Run ID: {run_id}{Colors.ENDC}")
                print(
                    f"{Colors.OKBLUE}   Silhouette Score: {best_run['metrics.silhouette_score']:.4f}{Colors.ENDC}"
                )
                print(
                    f"{Colors.OKBLUE}   Noise Ratio: {best_run['metrics.noise_ratio']:.4f}{Colors.ENDC}"
                )

                return run_id, best_run
            else:
                print(
                    f"{Colors.ORANGE}📝 No existing optimized model found for {dataset_name}{Colors.ENDC}"
                )
                return None, None

        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Error checking existing model: {e}{Colors.ENDC}")
            return None, None

    def load_optimized_params(self, run_id):
        """Load optimized parameters from MLflow run."""
        try:
            # Load parameters from the run
            params = mlflow.get_run(run_id).data.params

            # Convert string parameters back to appropriate types
            optimized_params = {
                "umap_n_neighbors": int(params["umap_n_neighbors"]),
                "umap_min_dist": float(params["umap_min_dist"]),
                "umap_n_components": int(params["umap_n_components"]),
                "hdbscan_min_cluster_size": int(params["hdbscan_min_cluster_size"]),
                "hdbscan_min_samples": int(params["hdbscan_min_samples"]),
                "hdbscan_cluster_selection_epsilon": float(
                    params["hdbscan_cluster_selection_epsilon"]
                ),
            }

            print(
                f"{Colors.OKGREEN}✅ Loaded optimized parameters from MLflow{Colors.ENDC}"
            )
            return optimized_params

        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Error loading parameters: {e}{Colors.ENDC}")
            return None

    def save_optimized_model(self, run_id, dataset_name, params, evaluation):
        """Save optimized model and parameters to MLflow."""
        try:
            # Log parameters to the current active run
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metric("silhouette_score", evaluation["silhouette_score"])
            mlflow.log_metric("noise_ratio", evaluation["noise_ratio"])
            mlflow.log_metric("n_clusters", evaluation["n_clusters"])

            # Log tags
            mlflow.set_tag("dataset_name", dataset_name)
            mlflow.set_tag("status", "completed")
            mlflow.set_tag("model_type", "umap_hdbscan")
            mlflow.set_tag("embedding_model", self.embedding_model_id)

            print(f"{Colors.OKGREEN}✅ Saved optimized model to MLflow{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Error saving model to MLflow: {e}{Colors.ENDC}")

    def start_mlflow_ui(self, port=5000):
        """Start MLflow UI server."""
        try:
            import subprocess
            import threading
            import time

            def run_mlflow_ui():
                try:
                    subprocess.run(
                        [
                            "mlflow",
                            "ui",
                            "--backend-store-uri",
                            self.mlflow_tracking_uri,
                            "--port",
                            str(port),
                            "--host",
                            "0.0.0.0",
                        ],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"{Colors.FAIL}❌ Failed to start MLflow UI: {e}{Colors.ENDC}"
                    )
                except FileNotFoundError:
                    print(
                        f"{Colors.FAIL}❌ MLflow not found. Please install it: pip install mlflow{Colors.ENDC}"
                    )

            # Start MLflow UI in a separate thread
            ui_thread = threading.Thread(target=run_mlflow_ui, daemon=True)
            ui_thread.start()

            # Wait a moment for the server to start
            time.sleep(2)

            print(f"{Colors.OKGREEN}✅ MLflow UI started successfully!{Colors.ENDC}")
            print(
                f"{Colors.OKBLUE}🌐 Open your browser and go to: http://localhost:{port}{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}📊 You can view experiments, runs, and model parameters{Colors.ENDC}"
            )
            print(f"{Colors.WARNING}⚠️  Press Ctrl+C to stop the UI server{Colors.ENDC}")

            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"{Colors.OKGREEN}✅ MLflow UI stopped{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.FAIL}❌ Error starting MLflow UI: {e}{Colors.ENDC}")

    def preprocess_texts(self, texts, min_length=20):
        """Filter out very short descriptions that might be noise."""
        filtered_texts = []
        filtered_indices = []

        for i, text in enumerate(texts):
            processed_text = str(text).strip()
            if len(processed_text) >= min_length:
                filtered_texts.append(processed_text)
                filtered_indices.append(i)

        return filtered_texts, filtered_indices

    def normalize_texts(self, texts):
        """Normalize text length to reduce variance while preserving meaning."""
        normalized = []
        for text in texts:
            # Only truncate extremely long texts (preserve more content)
            if len(text) > 500:  # Increased from 200 to 500
                text = text[:500] + "..."
            # Only pad very short texts
            elif len(text) < 30:  # Reduced from 50 to 30
                text = text + " " * (30 - len(text))
            normalized.append(text)
        return normalized

    def load_data(self):
        """Load CSV files and configuration."""
        print(
            f"{Colors.OKCYAN}{Colors.BOLD}📊 Loading data and configuration...{Colors.ENDC}"
        )

        # Load CSV files
        try:
            self.data_issues = pd.read_csv("data_issues.csv")
            self.issue_summaries = pd.read_csv("issue_summaries.csv")
            print(f"{Colors.OKGREEN}✅ CSV files loaded successfully!{Colors.ENDC}")
            print(
                f"{Colors.LIGHT_BLUE}   📋 Data Issues: {Colors.BOLD}{len(self.data_issues)}{Colors.ENDC}{Colors.LIGHT_BLUE} records{Colors.ENDC}"
            )
            print(
                f"{Colors.LIGHT_BLUE}   📋 Issue Summaries: {Colors.BOLD}{len(self.issue_summaries)}{Colors.ENDC}{Colors.LIGHT_BLUE} records{Colors.ENDC}"
            )
        except FileNotFoundError as e:
            print(f"{Colors.FAIL}❌ Error loading CSV files: {e}{Colors.ENDC}")
            raise

        # Load configuration
        try:
            with open("config/config.yaml", "r") as file:
                self.config = yaml.safe_load(file)
            self.embedding_model_id = self.config["aws"]["bedrock"][
                "embedding_model_id"
            ]
            self.llama_model_id = self.config["aws"]["bedrock"]["model_id"]
            print(
                f"{Colors.OKGREEN}✅ Configuration loaded: {Colors.BOLD}{self.embedding_model_id}{Colors.ENDC}"
            )
            print(
                f"{Colors.OKGREEN}✅ Llama model configured: {Colors.BOLD}{self.llama_model_id}{Colors.ENDC}"
            )
        except FileNotFoundError:
            print(
                f"{Colors.WARNING}⚠️  Config file not found. Using default configuration.{Colors.ENDC}"
            )
            self.config = {
                "aws": {
                    "region": "us-east-1",
                    "bedrock": {"embedding_model_id": "amazon.titan-embed-text-v2:0"},
                }
            }
            self.embedding_model_id = "amazon.titan-embed-text-v2:0"

        # Setup AWS Bedrock
        try:
            self.bedrock_client = boto3.client(
                "bedrock-runtime", region_name=self.config["aws"]["region"]
            )
            print(f"{Colors.OKGREEN}✅ AWS Bedrock client created{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error creating Bedrock client: {e}{Colors.ENDC}")
            raise

    def setup_vector_database(self):
        """Setup ChromaDB vector database."""
        print(
            f"{Colors.OKCYAN}{Colors.BOLD}🗄️ Setting up vector database...{Colors.ENDC}"
        )

        try:
            # Initialize ChromaDB with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Create collections without embedding function - we'll add embeddings manually
            self.data_issues_collection = self.chroma_client.get_or_create_collection(
                name="data_issues_embeddings",
                metadata={"description": "Embeddings for data issues dataset"},
                embedding_function=None,
            )

            self.issue_summaries_collection = (
                self.chroma_client.get_or_create_collection(
                    name="issue_summaries_embeddings",
                    metadata={"description": "Embeddings for issue summaries dataset"},
                    embedding_function=None,
                )
            )

            # Check for dimension mismatch and fix if needed
            self.check_and_fix_dimension_mismatch()

            # Check existing cache status
            self.check_cache_status()

            # Only run cache test if collections are empty (to avoid conflicts)
            if (
                self.data_issues_collection.count() == 0
                and self.issue_summaries_collection.count() == 0
            ):
                print(
                    f"{Colors.ORANGE}🧪 Running cache functionality test...{Colors.ENDC}"
                )
                cache_test_passed = self.test_cache_simple()
                if cache_test_passed:
                    print(
                        f"{Colors.OKGREEN}✅ Cache functionality verified{Colors.ENDC}"
                    )
                else:
                    print(
                        f"{Colors.FAIL}❌ Cache functionality test failed - this may cause issues{Colors.ENDC}"
                    )
            else:
                print(
                    f"{Colors.OKGREEN}✅ Cache contains existing data - skipping test to avoid conflicts{Colors.ENDC}"
                )

            print(f"{Colors.OKGREEN}✅ Vector database setup completed{Colors.ENDC}")

        except Exception as e:
            print(f"{Colors.FAIL}❌ Error setting up vector database: {e}{Colors.ENDC}")
            raise

    def check_and_fix_dimension_mismatch(self):
        """Check if cached embeddings have the correct dimension and reset if needed."""
        try:
            # Check if collections have data
            data_issues_count = self.data_issues_collection.count()
            issue_summaries_count = self.issue_summaries_collection.count()

            if data_issues_count == 0 and issue_summaries_count == 0:
                print(
                    f"{Colors.OKGREEN}   ✅ No cached data - no dimension check needed{Colors.ENDC}"
                )
                return

            # Check dimension of existing embeddings
            expected_dimension = 1024  # Titan embedding dimension

            if data_issues_count > 0:
                try:
                    sample_data = self.data_issues_collection.get(
                        limit=1, include=["embeddings"]
                    )
                    if (
                        sample_data.get("embeddings") is not None
                        and len(sample_data["embeddings"]) > 0
                    ):
                        cached_dimension = len(sample_data["embeddings"][0])
                        if cached_dimension != expected_dimension:
                            print(
                                f"{Colors.WARNING}   ⚠️  Dimension mismatch detected!{Colors.ENDC}"
                            )
                            print(
                                f"{Colors.WARNING}      • Expected: {expected_dimension} dimensions{Colors.ENDC}"
                            )
                            print(
                                f"{Colors.WARNING}      • Cached: {cached_dimension} dimensions{Colors.ENDC}"
                            )
                            print(
                                f"{Colors.WARNING}      • This will cause cache lookup failures{Colors.ENDC}"
                            )

                            # Reset the database
                            print(
                                f"{Colors.ORANGE}   🔄 Resetting vector database to fix dimension mismatch...{Colors.ENDC}"
                            )
                            self.reset_vector_database()
                            return
                except Exception as e:
                    print(
                        f"{Colors.WARNING}   ⚠️  Could not check data issues dimension: {e}{Colors.ENDC}"
                    )

            if issue_summaries_count > 0:
                try:
                    sample_data = self.issue_summaries_collection.get(
                        limit=1, include=["embeddings"]
                    )
                    if (
                        sample_data.get("embeddings") is not None
                        and len(sample_data["embeddings"]) > 0
                    ):
                        cached_dimension = len(sample_data["embeddings"][0])
                        if cached_dimension != expected_dimension:
                            print(
                                f"{Colors.WARNING}   ⚠️  Dimension mismatch detected!{Colors.ENDC}"
                            )
                            print(
                                f"{Colors.WARNING}      • Expected: {expected_dimension} dimensions{Colors.ENDC}"
                            )
                            print(
                                f"{Colors.WARNING}      • Cached: {cached_dimension} dimensions{Colors.ENDC}"
                            )
                            print(
                                f"{Colors.WARNING}      • This will cause cache lookup failures{Colors.ENDC}"
                            )

                            # Reset the database
                            print(
                                f"{Colors.ORANGE}   🔄 Resetting vector database to fix dimension mismatch...{Colors.ENDC}"
                            )
                            self.reset_vector_database()
                            return
                except Exception as e:
                    print(
                        f"{Colors.WARNING}   ⚠️  Could not check issue summaries dimension: {e}{Colors.ENDC}"
                    )

            print(
                f"{Colors.OKGREEN}   ✅ Dimension check passed - cache is compatible{Colors.ENDC}"
            )

        except Exception as e:
            print(
                f"{Colors.WARNING}   ⚠️  Error checking dimension mismatch: {e}{Colors.ENDC}"
            )

    def check_cache_status(self):
        """Check the status of cached embeddings."""
        try:
            data_issues_count = self.data_issues_collection.count()
            issue_summaries_count = self.issue_summaries_collection.count()

            print(f"{Colors.LIGHT_BLUE}   📊 Cache Status:{Colors.ENDC}")
            print(
                f"{Colors.LIGHT_BLUE}      • Data Issues: {data_issues_count} cached embeddings{Colors.ENDC}"
            )
            print(
                f"{Colors.LIGHT_BLUE}      • Issue Summaries: {issue_summaries_count} cached embeddings{Colors.ENDC}"
            )

            if data_issues_count > 0 or issue_summaries_count > 0:
                print(
                    f"{Colors.OKGREEN}      ✅ Found existing cached embeddings - will reduce AWS API calls!{Colors.ENDC}"
                )
                # Show sample of cached texts
                if data_issues_count > 0:
                    try:
                        sample = self.data_issues_collection.get(limit=1)
                        if sample["documents"]:
                            print(
                                f"{Colors.LIGHT_BLUE}      • Sample cached text: {sample['documents'][0][:50]}...{Colors.ENDC}"
                            )
                    except:
                        pass
            else:
                print(
                    f"{Colors.WARNING}      ⚠️  No cached embeddings found - will generate all embeddings{Colors.ENDC}"
                )

        except Exception as e:
            print(
                f"{Colors.WARNING}   ⚠️  Could not check cache status: {e}{Colors.ENDC}"
            )

    def generate_text_hash(self, text):
        """Generate hash for text to use as ID."""
        return hashlib.md5(text.encode()).hexdigest()

    def generate_titan_embeddings(self, texts, batch_size=10):
        """Generate embeddings using Amazon Titan."""
        print(
            f"{Colors.PURPLE}{Colors.BOLD}🎯 Generating Titan embeddings for {Colors.UNDERLINE}{len(texts)}{Colors.ENDC}{Colors.PURPLE} texts (AWS API calls)...{Colors.ENDC}"
        )

        # Estimate cost (rough estimate: $0.0001 per 1K tokens)
        total_chars = sum(len(text) for text in texts)
        estimated_cost = (total_chars / 1000) * 0.0001
        print(
            f"{Colors.WARNING}   💰 Estimated AWS cost: ~${estimated_cost:.4f}{Colors.ENDC}"
        )

        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i : i + batch_size]

            batch_embeddings = []
            for text in batch_texts:
                try:
                    # Prepare request body for single text
                    request_body = {"inputText": text}

                    # Call Bedrock API
                    response = self.bedrock_client.invoke_model(
                        modelId=self.embedding_model_id, body=json.dumps(request_body)
                    )

                    response_body = json.loads(response.get("body").read())
                    batch_embeddings.append(response_body["embedding"])

                except Exception as e:
                    print(
                        f"{Colors.FAIL}❌ Error embedding text: {text[:50]}... Error: {e}{Colors.ENDC}"
                    )
                    # Add zero embedding for failed text
                    batch_embeddings.append([0.0] * 1024)

            embeddings.extend(batch_embeddings)

        print(
            f"{Colors.OKGREEN}   ✅ Generated {len(embeddings)} embeddings successfully{Colors.ENDC}"
        )
        return np.array(embeddings)

    def get_embeddings_with_cache(self, texts, collection, dataset_name):
        """Get embeddings with caching from vector database - SIMPLIFIED AND ROBUST."""
        print(
            f"{Colors.OKCYAN}🔍 Getting embeddings for {dataset_name}...{Colors.ENDC}"
        )

        # First, let's check what's already in the cache
        try:
            all_cached = collection.get(
                include=["embeddings", "documents", "metadatas"]
            )
            cached_texts = {}

            # Simple validation and processing
            if (
                all_cached
                and isinstance(all_cached, dict)
                and "ids" in all_cached
                and "documents" in all_cached
                and "embeddings" in all_cached
                and len(all_cached["ids"]) > 0
                and len(all_cached["documents"]) > 0
                and len(all_cached["embeddings"]) > 0
            ):
                # Process cached embeddings
                for i in range(len(all_cached["ids"])):
                    try:
                        if (
                            i < len(all_cached["documents"])
                            and i < len(all_cached["embeddings"])
                            and all_cached["documents"][i]
                            and all_cached["embeddings"][i] is not None
                            and len(all_cached["embeddings"][i]) > 0
                        ):
                            cached_texts[all_cached["documents"][i]] = all_cached[
                                "embeddings"
                            ][i]
                    except:
                        continue

                print(f"   📊 Found {len(cached_texts)} cached embeddings")
            else:
                print(f"   📊 No cached embeddings found")
        except Exception as e:
            print(f"   ⚠️  Error checking cache: {e}")
            cached_texts = {}

        # Check which texts we need to generate
        missing_texts = []
        missing_indices = []
        existing_embeddings = {}

        for i, text in enumerate(texts):
            if text in cached_texts:
                # Found in cache!
                existing_embeddings[i] = np.array(cached_texts[text])
            else:
                # Not in cache, need to generate
                missing_texts.append(text)
                missing_indices.append(i)

        # Generate embeddings for missing texts
        if missing_texts:
            print(
                f"{Colors.WARNING}   📝 Generating {len(missing_texts)} new embeddings (cache miss)...{Colors.ENDC}"
            )
            new_embeddings = self.generate_titan_embeddings(missing_texts)

            # Store new embeddings in cache
            for i, (text, embedding) in enumerate(zip(missing_texts, new_embeddings)):
                text_hash = self.generate_text_hash(text)
                try:
                    collection.add(
                        embeddings=[embedding.tolist()],
                        documents=[text],
                        ids=[text_hash],
                        metadatas=[
                            {
                                "dataset": dataset_name,
                                "timestamp": datetime.now().isoformat(),
                                "text_hash": text_hash,
                            }
                        ],
                    )
                    existing_embeddings[missing_indices[i]] = embedding
                except Exception as e:
                    print(
                        f"{Colors.FAIL}   ❌ Failed to cache embedding for text {i}: {e}{Colors.ENDC}"
                    )
                    existing_embeddings[missing_indices[i]] = embedding
        else:
            print(
                f"{Colors.OKGREEN}   ✅ All {len(texts)} embeddings found in cache!{Colors.ENDC}"
            )

        # Return embeddings in original order
        result_embeddings = []
        for i in range(len(texts)):
            result_embeddings.append(existing_embeddings[i])

        # Calculate and display cost savings
        cache_hits = len(texts) - len(missing_texts)
        if cache_hits > 0:
            savings_percentage = (cache_hits / len(texts)) * 100
            print(
                f"{Colors.OKGREEN}   💰 Cost savings: {savings_percentage:.1f}% fewer API calls ({cache_hits}/{len(texts)} cached){Colors.ENDC}"
            )

        return np.array(result_embeddings)

    def clear_cache(self, dataset_name=None):
        """Clear the embedding cache."""
        if dataset_name == "data_issues" or dataset_name is None:
            try:
                self.data_issues_collection.delete(where={})
                print(f"{Colors.WARNING}🗑️  Cleared data_issues cache{Colors.ENDC}")
            except Exception as e:
                print(
                    f"{Colors.FAIL}❌ Error clearing data_issues cache: {e}{Colors.ENDC}"
                )

        if dataset_name == "issue_summaries" or dataset_name is None:
            try:
                self.issue_summaries_collection.delete(where={})
                print(f"{Colors.WARNING}🗑️  Cleared issue_summaries cache{Colors.ENDC}")
            except Exception as e:
                print(
                    f"{Colors.FAIL}❌ Error clearing issue_summaries cache: {e}{Colors.ENDC}"
                )

        if dataset_name is None:
            print(
                f"{Colors.WARNING}🗑️  All caches cleared - next run will generate all embeddings{Colors.ENDC}"
            )

    def reset_vector_database(self):
        """Completely reset the vector database to fix dimension issues."""
        print(
            f"{Colors.WARNING}⚠️  Resetting vector database to fix dimension issues...{Colors.ENDC}"
        )
        try:
            # Close existing client connections
            if hasattr(self, "chroma_client") and self.chroma_client:
                try:
                    self.chroma_client.reset()
                except:
                    pass

            # Force close any remaining connections
            import gc

            gc.collect()

            # Wait a moment for file handles to be released
            import time

            time.sleep(1)

            # Delete the entire ChromaDB directory with retry logic
            import shutil
            import os

            chroma_path = "./chroma_db"
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if os.path.exists(chroma_path):
                        shutil.rmtree(chroma_path)
                        print(
                            f"{Colors.OKGREEN}✅ Deleted existing ChromaDB directory{Colors.ENDC}"
                        )
                        break
                    else:
                        print(
                            f"{Colors.OKGREEN}✅ ChromaDB directory already removed{Colors.ENDC}"
                        )
                        break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        print(
                            f"{Colors.WARNING}⚠️  File lock detected, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries}){Colors.ENDC}"
                        )
                        time.sleep(2)
                    else:
                        print(
                            f"{Colors.FAIL}❌ Could not delete ChromaDB directory after {max_retries} attempts: {e}{Colors.ENDC}"
                        )
                        # Try to clear collections instead
                        try:
                            if hasattr(self, "data_issues_collection"):
                                self.data_issues_collection.delete(where={})
                            if hasattr(self, "issue_summaries_collection"):
                                self.issue_summaries_collection.delete(where={})
                            print(
                                f"{Colors.OKGREEN}✅ Cleared collections instead{Colors.ENDC}"
                            )
                        except:
                            pass
                        return False

            # Recreate the client and collections
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Create fresh collections without embedding function - we'll add embeddings manually
            self.data_issues_collection = self.chroma_client.get_or_create_collection(
                name="data_issues_embeddings",
                metadata={"description": "Embeddings for data issues dataset"},
                embedding_function=None,
            )

            self.issue_summaries_collection = (
                self.chroma_client.get_or_create_collection(
                    name="issue_summaries_embeddings",
                    metadata={"description": "Embeddings for issue summaries dataset"},
                    embedding_function=None,
                )
            )

            print(f"{Colors.OKGREEN}✅ Vector database reset completed{Colors.ENDC}")
            return True

        except Exception as e:
            print(f"{Colors.FAIL}❌ Error resetting vector database: {e}{Colors.ENDC}")
            return False

    def rebuild_cache(self, dataset_name=None):
        """Rebuild the embedding cache by clearing and regenerating."""
        print(
            f"{Colors.WARNING}🔄 Rebuilding cache for {dataset_name or 'all datasets'}...{Colors.ENDC}"
        )

        # Clear the cache first
        self.clear_cache(dataset_name)

        # Force regeneration by running the analysis again
        if dataset_name == "data_issues" or dataset_name is None:
            print(
                f"{Colors.OKCYAN}🔄 Regenerating data_issues embeddings...{Colors.ENDC}"
            )
            # This will be handled in the next run_clustering_analysis call

        if dataset_name == "issue_summaries" or dataset_name is None:
            print(
                f"{Colors.OKCYAN}🔄 Regenerating issue_summaries embeddings...{Colors.ENDC}"
            )
            # This will be handled in the next run_clustering_analysis call

    def validate_cache(self):
        """Validate that the cache is working properly."""
        print(f"{Colors.ORANGE}🔍 Validating cache functionality...{Colors.ENDC}")

        try:
            # Test data_issues cache
            if self.data_issues_collection.count() > 0:
                print(f"{Colors.OKCYAN}   Testing data_issues cache...{Colors.ENDC}")
                data_issues_working = self.test_cache_functionality(
                    self.data_issues_collection, "data_issues"
                )
            else:
                print(
                    f"{Colors.WARNING}   No data_issues embeddings in cache{Colors.ENDC}"
                )
                data_issues_working = True  # Not an error if empty

            # Test issue_summaries cache
            if self.issue_summaries_collection.count() > 0:
                print(
                    f"{Colors.OKCYAN}   Testing issue_summaries cache...{Colors.ENDC}"
                )
                issue_summaries_working = self.test_cache_functionality(
                    self.issue_summaries_collection, "issue_summaries"
                )
            else:
                print(
                    f"{Colors.WARNING}   No issue_summaries embeddings in cache{Colors.ENDC}"
                )
                issue_summaries_working = True  # Not an error if empty

            if data_issues_working and issue_summaries_working:
                print(f"{Colors.OKGREEN}✅ Cache validation passed{Colors.ENDC}")
                return True
            else:
                print(f"{Colors.FAIL}❌ Cache validation failed{Colors.ENDC}")
                return False

        except Exception as e:
            print(f"{Colors.FAIL}❌ Cache validation error: {e}{Colors.ENDC}")
            return False

    def test_cache_simple(self):
        """Simple cache test - add a test embedding and retrieve it."""
        print(f"{Colors.ORANGE}🧪 Running simple cache test...{Colors.ENDC}")

        test_text = "This is a test text for cache verification"
        test_hash = self.generate_text_hash(test_text)

        try:
            # Generate a real Titan embedding for the test
            test_embedding = self.generate_titan_embeddings([test_text])[0].tolist()

            self.data_issues_collection.add(
                embeddings=[test_embedding],
                documents=[test_text],
                ids=[test_hash],
                metadatas=[{"test": "true", "timestamp": datetime.now().isoformat()}],
            )
            print(
                f"{Colors.OKGREEN}   ✅ Successfully added test embedding{Colors.ENDC}"
            )

            # Try to retrieve it
            results = self.data_issues_collection.get(ids=[test_hash])
            if results["ids"] and len(results["ids"]) > 0:
                print(
                    f"{Colors.OKGREEN}   ✅ Successfully retrieved test embedding{Colors.ENDC}"
                )

                # Clean up - remove test embedding
                self.data_issues_collection.delete(ids=[test_hash])
                print(
                    f"{Colors.OKGREEN}   ✅ Cache test completed successfully{Colors.ENDC}"
                )
                return True
            else:
                print(
                    f"{Colors.FAIL}   ❌ Failed to retrieve test embedding{Colors.ENDC}"
                )
                return False

        except Exception as e:
            print(f"{Colors.FAIL}   ❌ Cache test failed: {e}{Colors.ENDC}")
            return False

    def objective(self, trial, embeddings):
        """Optuna objective function for clustering optimization with reduced noise."""
        # UMAP parameters - more relaxed for noise reduction
        n_neighbors = trial.suggest_int("umap_n_neighbors", 3, 15)  # Reduced range
        min_dist = trial.suggest_float("umap_min_dist", 0.0, 0.5)  # Reduced max
        n_components = trial.suggest_int("umap_n_components", 2, 8)  # Reduced max

        # HDBSCAN parameters - more relaxed for noise reduction
        min_cluster_size = trial.suggest_int(
            "hdbscan_min_cluster_size", 2, 6
        )  # Reduced max
        min_samples = trial.suggest_int("hdbscan_min_samples", 1, 3)  # Reduced max
        cluster_selection_epsilon = trial.suggest_float(
            "hdbscan_cluster_selection_epsilon", 0.0, 0.3
        )  # Reduced max

        try:
            # Apply UMAP
            umap_reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                random_state=42,
                metric="cosine",
            )
            umap_embeddings = umap_reducer.fit_transform(embeddings)

            # Apply HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric="euclidean",
            )
            cluster_labels = clusterer.fit_predict(umap_embeddings)

            # Calculate metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            if n_clusters < 2:
                return -1000

            # Silhouette score
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette = silhouette_score(
                    umap_embeddings[non_noise_mask], cluster_labels[non_noise_mask]
                )
            else:
                silhouette = -1

            # Combined score - reduced noise penalty
            noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
            score = silhouette - (
                noise_ratio * 0.5
            )  # Half the noise penalty for better clustering

            return score

        except Exception as e:
            return -1000

    def optimize_clustering(self, embeddings, name, n_trials=20):
        """Optimize clustering parameters with MLflow integration."""
        print(
            f"{Colors.OKCYAN}{Colors.BOLD}🔍 Optimizing clustering for {name}...{Colors.ENDC}"
        )

        # Check for existing optimized model
        run_id, best_run = self.check_existing_model(name)

        if run_id is not None:
            # Load existing optimized parameters
            optimized_params = self.load_optimized_params(run_id)
            if optimized_params is not None:
                print(
                    f"{Colors.OKGREEN}✅ Using existing optimized parameters from MLflow{Colors.ENDC}"
                )
                return optimized_params

        # No existing model found, run optimization
        print(
            f"{Colors.ORANGE}📝 Running new optimization with {n_trials} trials...{Colors.ENDC}"
        )

        # Start MLflow run for optimization
        with mlflow.start_run() as run:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self.objective(trial, embeddings), n_trials=n_trials
            )

            # Log optimization results
            mlflow.log_metric("best_score", study.best_value)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("dataset_name", name)

            print(f"{Colors.OKGREEN}✅ Best score: {study.best_value:.4f}{Colors.ENDC}")
            print(
                f"{Colors.OKBLUE}📊 Best parameters: {study.best_params}{Colors.ENDC}"
            )

            # Save the optimized model
            self.save_optimized_model(
                run.info.run_id,
                name,
                study.best_params,
                {
                    "silhouette_score": study.best_value,
                    "noise_ratio": 0.0,  # Will be calculated in apply_clustering
                    "n_clusters": 0,  # Will be calculated in apply_clustering
                },
            )

            return study.best_params

    def apply_clustering(self, embeddings, params, name):
        """Apply UMAP and HDBSCAN clustering."""
        print(
            f"{Colors.OKCYAN}{Colors.BOLD}🚀 Applying clustering for {name}...{Colors.ENDC}"
        )

        # Apply UMAP
        umap_reducer = umap.UMAP(
            n_neighbors=params["umap_n_neighbors"],
            min_dist=params["umap_min_dist"],
            n_components=params["umap_n_components"],
            random_state=42,
            metric="cosine",
        )
        umap_embeddings = umap_reducer.fit_transform(embeddings)

        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["hdbscan_min_cluster_size"],
            min_samples=params["hdbscan_min_samples"],
            cluster_selection_epsilon=params["hdbscan_cluster_selection_epsilon"],
            metric="euclidean",
        )
        cluster_labels = clusterer.fit_predict(umap_embeddings)

        return umap_embeddings, cluster_labels

    def evaluate_clustering(self, umap_embeddings, cluster_labels, name):
        """Evaluate clustering quality."""
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)

        # Calculate metrics
        non_noise_mask = cluster_labels != -1

        if np.sum(non_noise_mask) > 1 and n_clusters > 1:
            silhouette = silhouette_score(
                umap_embeddings[non_noise_mask], cluster_labels[non_noise_mask]
            )
        else:
            silhouette = -1

        print(f"\n📊 {name} Clustering Results:")
        print(f"   • Number of clusters: {n_clusters}")
        print(f"   • Noise ratio: {noise_ratio:.1%}")
        print(f"   • Silhouette score: {silhouette:.3f}")

        return {
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "silhouette_score": silhouette,
        }

    def run_clustering_analysis(self):
        """Run the complete clustering analysis with noise reduction."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
        print(
            f"{Colors.HEADER}{Colors.BOLD}🎯 RUNNING CLUSTERING ANALYSIS WITH NOISE REDUCTION{Colors.ENDC}"
        )
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")

        # Preprocess texts to reduce noise
        print(
            f"{Colors.ORANGE}{Colors.BOLD}🔧 Preprocessing texts to reduce noise...{Colors.ENDC}"
        )
        issue_texts = self.data_issues["description"].tolist()
        summary_texts = self.issue_summaries["summary"].tolist()

        # Filter very short descriptions (less aggressive)
        issue_texts, self.issue_indices = self.preprocess_texts(
            issue_texts, min_length=10  # Reduced from 20 to 10
        )
        summary_texts, self.summary_indices = self.preprocess_texts(
            summary_texts, min_length=10  # Reduced from 20 to 10
        )

        print(
            f"{Colors.LIGHT_GREEN}📝 Filtered to {Colors.BOLD}{len(issue_texts)}{Colors.ENDC}{Colors.LIGHT_GREEN} data issues (from {len(self.data_issues)}){Colors.ENDC}"
        )
        print(
            f"{Colors.LIGHT_GREEN}📝 Filtered to {Colors.BOLD}{len(summary_texts)}{Colors.ENDC}{Colors.LIGHT_GREEN} summaries (from {len(self.issue_summaries)}){Colors.ENDC}"
        )

        # Get embeddings with caching for original texts (before normalization)
        self.issue_embeddings = self.get_embeddings_with_cache(
            issue_texts, self.data_issues_collection, "data_issues"
        )

        self.summary_embeddings = self.get_embeddings_with_cache(
            summary_texts, self.issue_summaries_collection, "issue_summaries"
        )

        # Normalize text lengths after getting embeddings
        issue_texts = self.normalize_texts(issue_texts)
        summary_texts = self.normalize_texts(summary_texts)

        # Optimize clustering parameters
        print(
            f"\n{Colors.OKBLUE}{Colors.BOLD}🔍 Optimizing clustering parameters...{Colors.ENDC}"
        )
        self.issue_best_params = self.optimize_clustering(
            self.issue_embeddings, "Data Issues"
        )
        self.summary_best_params = self.optimize_clustering(
            self.summary_embeddings, "Issue Summaries"
        )

        # Apply clustering
        print(
            f"\n{Colors.PINK}{Colors.BOLD}🚀 Applying optimized clustering...{Colors.ENDC}"
        )
        self.issue_umap, self.issue_clusters = self.apply_clustering(
            self.issue_embeddings, self.issue_best_params, "Data Issues"
        )
        self.summary_umap, self.summary_clusters = self.apply_clustering(
            self.summary_embeddings, self.summary_best_params, "Issue Summaries"
        )

        # Evaluate results
        print(
            f"\n{Colors.LIGHT_YELLOW}{Colors.BOLD}📈 Evaluating clustering quality...{Colors.ENDC}"
        )
        self.issue_evaluation = self.evaluate_clustering(
            self.issue_umap, self.issue_clusters, "Data Issues"
        )
        self.summary_evaluation = self.evaluate_clustering(
            self.summary_umap, self.summary_clusters, "Issue Summaries"
        )

        # Save results
        self.save_clustering_results()

        # Show preprocessing statistics
        self.show_preprocessing_stats()

        print(
            f"{Colors.OKGREEN}{Colors.BOLD}✅ Clustering analysis completed!{Colors.ENDC}"
        )

        # Return the dataframes and clusters for the complete analysis
        data_issues_df = (
            self.data_issues.iloc[self.issue_indices].copy()
            if self.issue_indices is not None
            else self.data_issues.copy()
        )
        issue_summaries_df = (
            self.issue_summaries.iloc[self.summary_indices].copy()
            if self.summary_indices is not None
            else self.issue_summaries.copy()
        )

        return (
            data_issues_df,
            issue_summaries_df,
            self.issue_clusters,
            self.summary_clusters,
        )

    def save_clustering_results(self):
        """Save clustering results to files."""
        print("💾 Saving clustering results...")

        # Create results directory
        os.makedirs("complete_analysis_results", exist_ok=True)

        # Save numpy arrays
        np.save("complete_analysis_results/issue_embeddings.npy", self.issue_embeddings)
        np.save(
            "complete_analysis_results/summary_embeddings.npy", self.summary_embeddings
        )
        np.save("complete_analysis_results/issue_umap.npy", self.issue_umap)
        np.save("complete_analysis_results/issue_clusters.npy", self.issue_clusters)
        np.save("complete_analysis_results/summary_umap.npy", self.summary_umap)
        np.save("complete_analysis_results/summary_clusters.npy", self.summary_clusters)

        # Save results as JSON
        with open("complete_analysis_results/clustering_results.json", "w") as f:
            json.dump(
                {
                    "data_issues_evaluation": self.issue_evaluation,
                    "issue_summaries_evaluation": self.summary_evaluation,
                    "data_issues_params": self.issue_best_params,
                    "issue_summaries_params": self.summary_best_params,
                    "embedding_model_id": self.embedding_model_id,
                    "aws_region": self.config["aws"]["region"],
                },
                f,
                indent=2,
            )

        print("✅ Clustering results saved!")

    def show_preprocessing_stats(self):
        """Show statistics about the preprocessing steps."""
        if self.issue_indices is not None and self.summary_indices is not None:
            print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")

            print(
                f"{Colors.HEADER}{Colors.BOLD}🔧 PREPROCESSING STATISTICS{Colors.ENDC}"
            )
            print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")

            original_issues = len(self.data_issues)
            filtered_issues = len(self.issue_indices)
            original_summaries = len(self.issue_summaries)
            filtered_summaries = len(self.summary_indices)

            print(f"📊 Data Issues:")
            print(f"   • Original: {original_issues} records")
            print(f"   • After filtering: {filtered_issues} records")
            print(
                f"   • Removed: {original_issues - filtered_issues} short descriptions"
            )
            print(f"   • Retention rate: {(filtered_issues/original_issues)*100:.1f}%")

            print(f"\n📊 Issue Summaries:")
            print(f"   • Original: {original_summaries} records")
            print(f"   • After filtering: {filtered_summaries} records")
            print(
                f"   • Removed: {original_summaries - filtered_summaries} short descriptions"
            )
            print(
                f"   • Retention rate: {(filtered_summaries/original_summaries)*100:.1f}%"
            )

            print(f"\n🎯 Expected Noise Reduction:")
            print(
                f"   • Removed {(original_issues + original_summaries) - (filtered_issues + filtered_summaries)} potential noise points"
            )
            print(
                f"   • This should significantly reduce the noise ratio in clustering"
            )

    def extract_keywords(self, texts, max_features=100):
        """Extract keywords from a list of texts using TF-IDF."""
        # Clean texts
        cleaned_texts = []
        for text in texts:
            cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
            cleaned_texts.append(cleaned)

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Calculate average TF-IDF scores for each feature
            avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)

            # Create keyword dictionary
            keywords = {}
            for i, feature in enumerate(feature_names):
                if avg_scores[i] > 0:
                    keywords[feature] = avg_scores[i]

            return dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            print(f"⚠️  Warning: Could not extract keywords: {e}")
            return {}

    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts using TF-IDF and cosine similarity."""
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0

    def compare_clusters(
        self,
        data_issues_df,
        issue_summaries_df,
        data_issues_clusters,
        issue_summaries_clusters,
        similarity_threshold=0.3,
    ):
        """Compare clusters between the two datasets."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}🔍 RUNNING CLUSTER COMPARISON{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")

        print(f"{Colors.OKCYAN}🔍 Comparing clusters between datasets...{Colors.ENDC}")

        # Get cluster texts from the provided dataframes and clusters
        data_issues_clusters_dict = defaultdict(list)
        issue_summaries_clusters_dict = defaultdict(list)

        # Group texts by cluster ID
        for i, cluster_id in enumerate(data_issues_clusters):
            data_issues_clusters_dict[cluster_id].append(
                data_issues_df["description"].iloc[i]
            )

        for i, cluster_id in enumerate(issue_summaries_clusters):
            issue_summaries_clusters_dict[cluster_id].append(
                issue_summaries_df["summary"].iloc[i]
            )

        # Extract keywords for each cluster
        data_issues_keywords = {}
        issue_summaries_keywords = {}

        for cluster_id, texts in data_issues_clusters_dict.items():
            if cluster_id != -1:  # Skip noise cluster
                data_issues_keywords[cluster_id] = self.extract_keywords(texts)

        for cluster_id, texts in issue_summaries_clusters_dict.items():
            if cluster_id != -1:  # Skip noise cluster
                issue_summaries_keywords[cluster_id] = self.extract_keywords(texts)

        # Find similar clusters
        similar_clusters = []
        unique_data_issues_clusters = []
        unique_issue_summaries_clusters = []

        # Compare each pair of clusters
        for di_cluster_id, di_texts in data_issues_clusters_dict.items():
            if di_cluster_id == -1:  # Skip noise cluster
                continue

            best_similarity = 0
            best_match = None

            for is_cluster_id, is_texts in issue_summaries_clusters_dict.items():
                if is_cluster_id == -1:  # Skip noise cluster
                    continue

                # Calculate similarity between cluster representatives
                di_representative = " ".join(
                    di_texts[:3]
                )  # Use first 3 texts as representative
                is_representative = " ".join(is_texts[:3])

                similarity = self.calculate_text_similarity(
                    di_representative, is_representative
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = is_cluster_id

            if best_similarity >= similarity_threshold:
                similar_clusters.append(
                    {
                        "data_issues_cluster": di_cluster_id,
                        "issue_summaries_cluster": best_match,
                        "similarity_score": best_similarity,
                        "data_issues_texts": di_texts,
                        "issue_summaries_texts": issue_summaries_clusters_dict[
                            best_match
                        ],
                        "data_issues_keywords": data_issues_keywords.get(
                            di_cluster_id, {}
                        ),
                        "issue_summaries_keywords": issue_summaries_keywords.get(
                            best_match, {}
                        ),
                    }
                )
            else:
                unique_data_issues_clusters.append(
                    {
                        "cluster_id": di_cluster_id,
                        "texts": di_texts,
                        "keywords": data_issues_keywords.get(di_cluster_id, {}),
                    }
                )

        # Find unique issue_summaries clusters
        matched_is_clusters = {
            match["issue_summaries_cluster"] for match in similar_clusters
        }
        for is_cluster_id, is_texts in issue_summaries_clusters_dict.items():
            if is_cluster_id != -1 and is_cluster_id not in matched_is_clusters:
                unique_issue_summaries_clusters.append(
                    {
                        "cluster_id": is_cluster_id,
                        "texts": is_texts,
                        "keywords": issue_summaries_keywords.get(is_cluster_id, {}),
                    }
                )

        # Store results
        self.comparison_results = {
            "similar_clusters": similar_clusters,
            "unique_data_issues_clusters": unique_data_issues_clusters,
            "unique_issue_summaries_clusters": unique_issue_summaries_clusters,
            "similarity_threshold": similarity_threshold,
            "summary": {
                "total_similar_clusters": len(similar_clusters),
                "total_unique_data_issues_clusters": len(unique_data_issues_clusters),
                "total_unique_issue_summaries_clusters": len(
                    unique_issue_summaries_clusters
                ),
                "total_data_issues_clusters": len(similar_clusters)
                + len(unique_data_issues_clusters),
                "total_issue_summaries_clusters": len(similar_clusters)
                + len(unique_issue_summaries_clusters),
            },
        }

        print(f"{Colors.OKGREEN}✅ Cluster comparison completed!{Colors.ENDC}")

        # Debug: Show actual cluster counts from clustering algorithm
        print(f"\n{Colors.LIGHT_YELLOW}🔍 DEBUG - Actual Cluster Counts:{Colors.ENDC}")
        print(
            f"   • Data Issues clusters (from algorithm): {len(data_issues_clusters_dict) - 1} (excluding noise)"
        )
        print(
            f"   • Issue Summaries clusters (from algorithm): {len(issue_summaries_clusters_dict) - 1} (excluding noise)"
        )
        print(f"   • Similar clusters found: {len(similar_clusters)}")
        print(f"   • Unique data issues clusters: {len(unique_data_issues_clusters)}")
        print(
            f"   • Unique issue summaries clusters: {len(unique_issue_summaries_clusters)}"
        )

        # Show cluster sizes for better understanding
        print(f"\n{Colors.LIGHT_YELLOW}🔍 CLUSTER SIZE ANALYSIS:{Colors.ENDC}")
        for cluster_id, texts in data_issues_clusters_dict.items():
            if cluster_id != -1:
                print(f"   • Data Issues Cluster {cluster_id}: {len(texts)} records")
        for cluster_id, texts in issue_summaries_clusters_dict.items():
            if cluster_id != -1:
                print(
                    f"   • Issue Summaries Cluster {cluster_id}: {len(texts)} records"
                )

        # Convert similar_clusters to the format expected by run_complete_analysis
        similar_clusters_formatted = []
        for cluster_info in similar_clusters:
            similar_clusters_formatted.append(
                (
                    cluster_info["data_issues_cluster"],
                    cluster_info["issue_summaries_cluster"],
                    cluster_info["similarity_score"],
                )
            )

        return (
            similar_clusters_formatted,
            unique_data_issues_clusters,
            unique_issue_summaries_clusters,
        )

    def print_comparison_summary(
        self,
        similar_clusters,
        unique_data_issues_clusters,
        unique_issue_summaries_clusters,
    ):
        """Print a summary of the cluster comparison results."""
        total_similar_clusters = len(similar_clusters)
        total_unique_data_issues_clusters = len(unique_data_issues_clusters)
        total_unique_issue_summaries_clusters = len(unique_issue_summaries_clusters)
        total_data_issues_clusters = (
            total_similar_clusters + total_unique_data_issues_clusters
        )
        total_issue_summaries_clusters = (
            total_similar_clusters + total_unique_issue_summaries_clusters
        )

        print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
        print(
            f"{Colors.HEADER}{Colors.BOLD}📊 FINAL CLUSTER COMPARISON SUMMARY{Colors.ENDC}"
        )
        print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")

        print(
            f"\n{Colors.LIGHT_BLUE}📊 Similar Clusters: {Colors.BOLD}{total_similar_clusters}{Colors.ENDC}"
        )
        print(
            f"{Colors.LIGHT_BLUE}📊 Unique Data Issues Clusters: {Colors.BOLD}{total_unique_data_issues_clusters}{Colors.ENDC}"
        )
        print(
            f"{Colors.LIGHT_BLUE}📊 Unique Issue Summaries Clusters: {Colors.BOLD}{total_unique_issue_summaries_clusters}{Colors.ENDC}"
        )
        print(
            f"{Colors.LIGHT_BLUE}📊 Total Data Issues Clusters: {Colors.BOLD}{total_data_issues_clusters}{Colors.ENDC}"
        )
        print(
            f"{Colors.LIGHT_BLUE}📊 Total Issue Summaries Clusters: {Colors.BOLD}{total_issue_summaries_clusters}{Colors.ENDC}"
        )

        # Calculate overlap percentage
        total_clusters = total_data_issues_clusters + total_issue_summaries_clusters
        overlap_percentage = (total_similar_clusters * 2 / total_clusters) * 100

        print(
            f"\n{Colors.OKGREEN}🎯 Cluster Overlap: {Colors.BOLD}{overlap_percentage:.1f}%{Colors.ENDC}"
        )

        # Add verification of the math
        print(f"\n{Colors.ORANGE}🔍 VERIFICATION:{Colors.ENDC}")
        print(
            f"   • Data Issues: {total_similar_clusters} similar + {total_unique_data_issues_clusters} unique = {total_data_issues_clusters} total (from comparison)"
        )
        print(
            f"   • Issue Summaries: {total_similar_clusters} similar + {total_unique_issue_summaries_clusters} unique = {total_issue_summaries_clusters} total (from comparison)"
        )

        print(f"\n{Colors.LIGHT_GREEN}💡 EXPLANATION:{Colors.ENDC}")
        print(
            f"   • The 'total' is the sum of similar + unique clusters found by the comparison logic"
        )
        print(
            f"   • This may differ from the actual number of clusters found by HDBSCAN"
        )
        print(
            f"   • Having 100 records doesn't guarantee 100 clusters - HDBSCAN groups similar items together"
        )

    def print_similar_clusters(self, max_examples=3):
        """Print details of similar clusters."""
        if (
            not self.comparison_results
            or not self.comparison_results["similar_clusters"]
        ):
            print("❌ No similar clusters found.")
            return

        print(
            f"\n🔗 SIMILAR CLUSTERS (showing up to {max_examples} examples per cluster):"
        )
        print("=" * 80)

        for i, match in enumerate(self.comparison_results["similar_clusters"], 1):
            print(
                f"\n{i}. Data Issues Cluster {match['data_issues_cluster']} ↔ Issue Summaries Cluster {match['issue_summaries_cluster']}"
            )
            print(f"   Similarity Score: {match['similarity_score']:.3f}")
            print(f"   📋 Data Issues Examples:")
            for j, text in enumerate(match["data_issues_texts"][:max_examples], 1):
                print(f"      {j}. {text[:100]}{'...' if len(text) > 100 else ''}")

            print(f"   📋 Issue Summaries Examples:")
            for j, text in enumerate(match["issue_summaries_texts"][:max_examples], 1):
                print(f"      {j}. {text[:100]}{'...' if len(text) > 100 else ''}")

            # Show common keywords
            di_keywords = set(match["data_issues_keywords"].keys())
            is_keywords = set(match["issue_summaries_keywords"].keys())
            common_keywords = di_keywords.intersection(is_keywords)

            if common_keywords:
                print(f"   🔑 Common Keywords: {', '.join(list(common_keywords)[:5])}")

    def save_comparison_results(self):
        """Save comparison results to files."""
        if not self.comparison_results:
            print("❌ No comparison results available. Run compare_clusters() first.")
            return

        print("💾 Saving comparison results...")

        # Create results directory
        os.makedirs("complete_analysis_results", exist_ok=True)

        # Save detailed results as JSON
        with open("complete_analysis_results/comparison_results.json", "w") as f:
            json.dump(self.comparison_results, f, indent=2, default=str)

        # Save summary as CSV
        summary_data = []

        # Similar clusters
        for match in self.comparison_results["similar_clusters"]:
            summary_data.append(
                {
                    "cluster_type": "similar",
                    "data_issues_cluster": match["data_issues_cluster"],
                    "issue_summaries_cluster": match["issue_summaries_cluster"],
                    "similarity_score": match["similarity_score"],
                    "data_issues_size": len(match["data_issues_texts"]),
                    "issue_summaries_size": len(match["issue_summaries_texts"]),
                    "common_keywords": len(
                        set(match["data_issues_keywords"].keys())
                        & set(match["issue_summaries_keywords"].keys())
                    ),
                }
            )

        # Unique data issues clusters
        for cluster in self.comparison_results["unique_data_issues_clusters"]:
            summary_data.append(
                {
                    "cluster_type": "unique_data_issues",
                    "data_issues_cluster": cluster["cluster_id"],
                    "issue_summaries_cluster": None,
                    "similarity_score": None,
                    "data_issues_size": len(cluster["texts"]),
                    "issue_summaries_size": None,
                    "common_keywords": None,
                }
            )

        # Unique issue summaries clusters
        for cluster in self.comparison_results["unique_issue_summaries_clusters"]:
            summary_data.append(
                {
                    "cluster_type": "unique_issue_summaries",
                    "data_issues_cluster": None,
                    "issue_summaries_cluster": cluster["cluster_id"],
                    "similarity_score": None,
                    "data_issues_size": None,
                    "issue_summaries_size": len(cluster["texts"]),
                    "common_keywords": None,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            "complete_analysis_results/comparison_summary.csv", index=False
        )

        print("✅ Comparison results saved!")

    def generate_llama_classification(self, texts, cluster_id):
        """Generate zero-shot classification for a cluster using Llama model."""
        try:
            # Create context from all texts in the cluster
            context = "\n".join([f"- {text}" for text in texts])

            # Create the prompt for zero-shot classification
            # Following AWS Bedrock Llama prompt engineering best practices:
            # 1. Place the question/instruction at the end for best results
            # 2. Use clear, simple instructions
            # 3. Provide examples for better generalization
            prompt = f"""You are a data analyst tasked with identifying the main thematic category for a group of company issues.

Here are the issues in this cluster:
{context}

Based on the content above, what is the main thematic category or theme that describes these issues? 
Please provide only 1-3 words that best represent the common theme.

Examples of themes: "Data Quality", "Security Issues", "Performance Problems", "Network Issues", "User Access", "System Errors", "Compliance Issues", "Infrastructure Problems", "Data Loss", "Authentication Issues", "API Errors", "Database Issues", "Monitoring Problems", "Backup Issues", "Integration Problems"

Theme:"""

            # Use Llama 3 instruction format for AWS Bedrock
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

            # Prepare the request for Llama model using native structure
            # Following AWS Bedrock Llama 3 documentation

            native_request = {
                "prompt": formatted_prompt,
                "max_gen_len": self.config["aws"]["bedrock"]["max_tokens"],
                "temperature": self.config["aws"]["bedrock"]["temperature"],
                "top_p": 0.9,
            }
            request = json.dumps(native_request)

            # Call the Llama model via Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.llama_model_id, body=request
            )

            # Parse the response
            response_body = json.loads(response["body"].read())

            # Handle different response formats
            if "generation" in response_body:
                theme = response_body["generation"]
            elif "completion" in response_body:
                theme = response_body["completion"]
            elif "text" in response_body:
                theme = response_body["text"]
            else:
                print(
                    f"{Colors.WARNING}   ⚠️ Unexpected response format: {response_body}{Colors.ENDC}"
                )
                theme = "Unknown"

            # Clean up the theme (remove quotes, extra spaces, etc.)
            theme = theme.replace('"', "").replace("'", "").strip()

            # Extract just the theme from verbose responses
            # Look for patterns like "**Answer:** Theme" or just the theme
            if "**Answer:**" in theme:
                theme = theme.split("**Answer:**")[1].split("**")[0].strip()
            elif "Answer:" in theme:
                theme = theme.split("Answer:")[1].split("\n")[0].strip()

            # Remove any remaining markdown or formatting
            theme = theme.replace("*", "").replace("**", "").strip()

            # Limit to first few words (1-3 words as requested)
            words = theme.split()[:3]
            theme = " ".join(words)

            print(
                f"{Colors.OKGREEN}✅ Cluster {cluster_id} classified as: {Colors.BOLD}{theme}{Colors.ENDC}"
            )
            return theme

        except Exception as e:
            print(
                f"{Colors.FAIL}❌ Error classifying cluster {cluster_id}: {e}{Colors.ENDC}"
            )
            return f"Unknown-{cluster_id}"

    def generate_binary_classification(self, text):
        """Generate binary classification (Yes/No) for whether a text is data quality related."""
        try:
            # Create the prompt for binary classification
            prompt = f"""You are a data quality analyst. Determine if the following issue is related to data quality problems.

Issue: {text}

Is this issue related to data quality problems? Answer with ONLY "Yes" or "No".

Examples of data quality issues:
- Missing or incomplete data
- Duplicate records
- Data format inconsistencies
- Data validation errors
- Data accuracy problems
- Data completeness issues
- Data integrity violations
- Data standardization problems

Examples of non-data quality issues:
- Network connectivity problems
- System performance issues
- User access problems
- Security breaches
- Hardware failures
- Software bugs (not data-related)
- Infrastructure issues

Answer:"""

            # Use Llama 3 instruction format for AWS Bedrock
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"

            # Prepare the request for Llama model
            native_request = {
                "prompt": formatted_prompt,
                "max_gen_len": 5,  # Very short response for Yes/No
                "temperature": 0.0,  # Zero temperature for deterministic Yes/No answers
                "top_p": 0.1,
            }
            request = json.dumps(native_request)

            # Call the Llama model via Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.llama_model_id, body=request
            )

            # Parse the response
            response_body = json.loads(response["body"].read())

            # Handle different response formats
            if "generation" in response_body:
                answer = response_body["generation"]
            elif "completion" in response_body:
                answer = response_body["completion"]
            elif "text" in response_body:
                answer = response_body["text"]
            else:
                print(
                    f"{Colors.WARNING}   ⚠️ Unexpected response format: {response_body}{Colors.ENDC}"
                )
                return "Unknown"

            # Clean up the answer
            answer = answer.replace('"', "").replace("'", "").strip()

            # Remove any instruction tags that might be in the response
            answer = answer.replace("[/INST]", "").replace("[INST]", "").strip()

            # Extract just Yes/No from verbose responses
            if "**Answer:**" in answer:
                answer = answer.split("**Answer:**")[1].split("**")[0].strip()
            elif "Answer:" in answer:
                answer = answer.split("Answer:")[1].split("\n")[0].strip()

            # Normalize to Yes/No
            answer = answer.lower().strip()
            if answer.startswith("yes"):
                return "Yes"
            elif answer.startswith("no"):
                return "No"
            else:
                print(
                    f"{Colors.WARNING}   ⚠️ Unexpected binary answer: '{answer}'{Colors.ENDC}"
                )
                return "Unknown"

        except Exception as e:
            print(
                f"{Colors.FAIL}❌ Error in binary classification: {e}{Colors.ENDC}"
            )
            return "Unknown"

    def classify_all_records_binary(self, data_issues_df, issue_summaries_df):
        """Classify all individual records as Yes/No for data quality issues."""
        print(
            f"\n{Colors.HEADER}{Colors.BOLD}🔍 BINARY DATA QUALITY CLASSIFICATION{Colors.ENDC}"
        )
        print(f"{Colors.OKBLUE}{'=' * 60}{Colors.ENDC}")

        binary_results = {
            "data_issues": {},
            "issue_summaries": {}
        }

        # Classify data issues records
        print(f"{Colors.OKCYAN}📊 Classifying data issues records...{Colors.ENDC}")
        data_issues_classifications = []
        
        for idx, row in data_issues_df.iterrows():
            text = row['description']
            classification = self.generate_binary_classification(text)
            data_issues_classifications.append(classification)
            
            if idx < 5:  # Show first 5 classifications
                print(f"   Record {idx+1}: {Colors.BOLD}{classification}{Colors.ENDC}")
        
        # Add classification column to dataframe
        data_issues_df['data_quality_related'] = data_issues_classifications
        
        # Count classifications
        yes_count = data_issues_classifications.count("Yes")
        no_count = data_issues_classifications.count("No")
        unknown_count = data_issues_classifications.count("Unknown")
        
        print(f"\n{Colors.OKGREEN}📈 Data Issues Binary Classification Summary:{Colors.ENDC}")
        print(f"   ✅ Data Quality Related: {Colors.BOLD}{yes_count}{Colors.ENDC}")
        print(f"   ❌ Not Data Quality Related: {Colors.BOLD}{no_count}{Colors.ENDC}")
        print(f"   ❓ Unknown: {Colors.BOLD}{unknown_count}{Colors.ENDC}")
        print(f"   📊 Total Records: {Colors.BOLD}{len(data_issues_classifications)}{Colors.ENDC}")

        # Classify issue summaries records
        print(f"\n{Colors.OKCYAN}📊 Classifying issue summaries records...{Colors.ENDC}")
        issue_summaries_classifications = []
        
        for idx, row in issue_summaries_df.iterrows():
            text = row['summary']
            classification = self.generate_binary_classification(text)
            issue_summaries_classifications.append(classification)
            
            if idx < 5:  # Show first 5 classifications
                print(f"   Record {idx+1}: {Colors.BOLD}{classification}{Colors.ENDC}")
        
        # Add classification column to dataframe
        issue_summaries_df['data_quality_related'] = issue_summaries_classifications
        
        # Count classifications
        yes_count = issue_summaries_classifications.count("Yes")
        no_count = issue_summaries_classifications.count("No")
        unknown_count = issue_summaries_classifications.count("Unknown")
        
        print(f"\n{Colors.OKGREEN}📈 Issue Summaries Binary Classification Summary:{Colors.ENDC}")
        print(f"   ✅ Data Quality Related: {Colors.BOLD}{yes_count}{Colors.ENDC}")
        print(f"   ❌ Not Data Quality Related: {Colors.BOLD}{no_count}{Colors.ENDC}")
        print(f"   ❓ Unknown: {Colors.BOLD}{unknown_count}{Colors.ENDC}")
        print(f"   📊 Total Records: {Colors.BOLD}{len(issue_summaries_classifications)}{Colors.ENDC}")

        return data_issues_df, issue_summaries_df

    def classify_all_clusters(self):
        """Classify all clusters from both datasets using zero-shot classification."""
        print(
            f"\n{Colors.HEADER}{Colors.BOLD}🎯 ZERO-SHOT CLUSTER CLASSIFICATION{Colors.ENDC}"
        )
        print(f"{Colors.OKBLUE}{'=' * 60}{Colors.ENDC}")

        classification_results = {
            "data_issues_clusters": {},
            "issue_summaries_clusters": {},
        }

        # Classify data issues clusters
        if self.issue_clusters is not None:
            print(
                f"\n{Colors.OKCYAN}📊 Classifying Data Issues clusters...{Colors.ENDC}"
            )
            unique_clusters = set(self.issue_clusters)
            unique_clusters.discard(-1)  # Remove noise cluster

            for cluster_id in sorted(unique_clusters):
                # Get texts for this cluster
                cluster_texts = []
                for i, label in enumerate(self.issue_clusters):
                    if label == cluster_id:
                        cluster_texts.append(self.data_issues.iloc[i]["description"])

                if cluster_texts:
                    theme = self.generate_llama_classification(
                        cluster_texts, f"DI-{cluster_id}"
                    )
                    # Convert cluster_id to string for JSON serialization
                    classification_results["data_issues_clusters"][str(cluster_id)] = {
                        "theme": theme,
                        "texts": cluster_texts,
                        "size": len(cluster_texts),
                    }

        # Classify issue summaries clusters
        if self.summary_clusters is not None:
            print(
                f"\n{Colors.OKCYAN}📊 Classifying Issue Summaries clusters...{Colors.ENDC}"
            )
            unique_clusters = set(self.summary_clusters)
            unique_clusters.discard(-1)  # Remove noise cluster

            for cluster_id in sorted(unique_clusters):
                # Get texts for this cluster
                cluster_texts = []
                for i, label in enumerate(self.summary_clusters):
                    if label == cluster_id:
                        cluster_texts.append(self.issue_summaries.iloc[i]["summary"])

                if cluster_texts:
                    theme = self.generate_llama_classification(
                        cluster_texts, f"IS-{cluster_id}"
                    )
                    # Convert cluster_id to string for JSON serialization
                    classification_results["issue_summaries_clusters"][
                        str(cluster_id)
                    ] = {
                        "theme": theme,
                        "texts": cluster_texts,
                        "size": len(cluster_texts),
                    }

        # Store classification results
        self.classification_results = classification_results

        # Extract themes for the new format
        data_issues_themes = {}
        issue_summaries_themes = {}

        for cluster_id, info in classification_results["data_issues_clusters"].items():
            data_issues_themes[cluster_id] = info["theme"]

        for cluster_id, info in classification_results[
            "issue_summaries_clusters"
        ].items():
            issue_summaries_themes[cluster_id] = info["theme"]

        # Print summary
        self.print_classification_summary(data_issues_themes, issue_summaries_themes)

        # Save classification results
        self.save_classification_results(data_issues_themes, issue_summaries_themes)

        return data_issues_themes, issue_summaries_themes

    def print_classification_summary(self, data_issues_themes, issue_summaries_themes):
        """Print a summary of cluster classifications."""
        print(
            f"\n{Colors.HEADER}{Colors.BOLD}📋 CLUSTER CLASSIFICATION SUMMARY{Colors.ENDC}"
        )
        print(f"{Colors.OKBLUE}{'=' * 60}{Colors.ENDC}")

        # Data Issues clusters
        if data_issues_themes:
            print(f"\n{Colors.OKCYAN}📊 Data Issues Clusters:{Colors.ENDC}")
            for cluster_id, theme in data_issues_themes.items():
                print(
                    f"   {Colors.BOLD}Cluster {cluster_id}:{Colors.ENDC} {Colors.OKGREEN}{theme}{Colors.ENDC}"
                )

        # Issue Summaries clusters
        if issue_summaries_themes:
            print(f"\n{Colors.OKCYAN}📊 Issue Summaries Clusters:{Colors.ENDC}")
            for cluster_id, theme in issue_summaries_themes.items():
                print(
                    f"   {Colors.BOLD}Cluster {cluster_id}:{Colors.ENDC} {Colors.OKGREEN}{theme}{Colors.ENDC}"
                )

    def save_classification_results(self, data_issues_themes, issue_summaries_themes):
        """Save classification results to files."""
        print(f"{Colors.OKCYAN}💾 Saving classification results...{Colors.ENDC}")

        # Create results directory
        os.makedirs("complete_analysis_results", exist_ok=True)

        # Prepare classification results for saving
        classification_results = {
            "data_issues_clusters": data_issues_themes,
            "issue_summaries_clusters": issue_summaries_themes,
        }

        # Save detailed results as JSON
        with open("complete_analysis_results/cluster_classifications.json", "w") as f:
            json.dump(classification_results, f, indent=2, default=str)

        # Save summary as CSV
        summary_data = []

        # Data Issues clusters
        for cluster_id, theme in data_issues_themes.items():
            summary_data.append(
                {
                    "dataset": "data_issues",
                    "cluster_id": cluster_id,
                    "theme": theme,
                }
            )

        # Issue Summaries clusters
        for cluster_id, theme in issue_summaries_themes.items():
            summary_data.append(
                {
                    "dataset": "issue_summaries",
                    "cluster_id": cluster_id,
                    "theme": theme,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("complete_analysis_results/cluster_themes.csv", index=False)

        print(f"{Colors.OKGREEN}✅ Classification results saved!{Colors.ENDC}")
        print(
            f"{Colors.LIGHT_BLUE}📄 Detailed results: {Colors.BOLD}cluster_classifications.json{Colors.ENDC}"
        )
        print(
            f"{Colors.LIGHT_BLUE}📊 Summary: {Colors.BOLD}cluster_themes.csv{Colors.ENDC}"
        )

    def generate_cluster_hash(self, texts):
        """Generate a hash for a cluster based on its texts for caching."""
        # Sort texts to ensure consistent hash regardless of order
        sorted_texts = sorted([text.strip() for text in texts])
        text_content = "|||".join(sorted_texts)
        return hashlib.md5(text_content.encode()).hexdigest()

    def get_cached_classification(self, cluster_hash):
        """Get cached classification result if available."""
        cache_file = "complete_analysis_results/classification_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache = json.load(f)
                return cache.get(cluster_hash)
            except:
                return None
        return None

    def save_cached_classification(self, cluster_hash, theme):
        """Save classification result to cache."""
        cache_file = "complete_analysis_results/classification_cache.json"
        os.makedirs("complete_analysis_results", exist_ok=True)

        try:
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            else:
                cache = {}

            cache[cluster_hash] = theme

            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ Could not save to cache: {e}{Colors.ENDC}")

    def sample_representative_texts(self, texts, max_samples=10):
        """Sample representative texts from a cluster to reduce API calls."""
        if len(texts) <= max_samples:
            return texts

        # Simple strategy: take first, middle, and last texts
        # This could be improved with more sophisticated sampling
        n = len(texts)
        indices = []

        # Always include first and last
        indices.append(0)
        indices.append(n - 1)

        # Add middle texts
        if n > 2:
            step = max(1, (n - 2) // (max_samples - 2))
            for i in range(1, n - 1, step):
                if len(indices) < max_samples:
                    indices.append(i)

        return [texts[i] for i in sorted(indices)]

    def classify_cluster_optimized(self, cluster_data):
        """Optimized classification for a single cluster with caching and sampling."""
        cluster_id, texts, dataset_name = cluster_data

        # Generate hash for caching
        cluster_hash = self.generate_cluster_hash(texts)

        # Check cache first
        cached_theme = self.get_cached_classification(cluster_hash)
        if cached_theme:
            print(
                f"{Colors.OKGREEN}✅ Cache hit for cluster {cluster_id}: {Colors.BOLD}{cached_theme}{Colors.ENDC}"
            )
            return cluster_id, cached_theme, len(texts), True

        # Sample representative texts for large clusters
        if len(texts) > 20:
            sampled_texts = self.sample_representative_texts(texts, max_samples=15)
            print(
                f"{Colors.OKCYAN}📊 Cluster {cluster_id}: Using {len(sampled_texts)} representative texts from {len(texts)} total{Colors.ENDC}"
            )
        else:
            sampled_texts = texts

        # Generate classification
        theme = self.generate_llama_classification(sampled_texts, cluster_id)

        # Cache the result
        self.save_cached_classification(cluster_hash, theme)

        return cluster_id, theme, len(texts), False

    def classify_all_clusters_optimized(self, max_workers=4):
        """Optimized classification using parallel processing, caching, and sampling."""
        print(
            f"\n{Colors.HEADER}{Colors.BOLD}🚀 OPTIMIZED ZERO-SHOT CLUSTER CLASSIFICATION{Colors.ENDC}"
        )
        print(f"{Colors.OKBLUE}{'=' * 70}{Colors.ENDC}")
        print(
            f"{Colors.OKCYAN}⚡ Using parallel processing ({max_workers} workers), caching, and sampling{Colors.ENDC}"
        )

        classification_results = {
            "data_issues_clusters": {},
            "issue_summaries_clusters": {},
        }

        # Prepare all cluster data
        all_clusters = []

        # Data Issues clusters
        if self.issue_clusters is not None:
            unique_clusters = set(self.issue_clusters)
            unique_clusters.discard(-1)  # Remove noise cluster

            for cluster_id in sorted(unique_clusters):
                cluster_texts = []
                for i, label in enumerate(self.issue_clusters):
                    if label == cluster_id:
                        cluster_texts.append(self.data_issues.iloc[i]["description"])

                if cluster_texts:
                    all_clusters.append(
                        (f"DI-{cluster_id}", cluster_texts, "data_issues")
                    )

        # Issue Summaries clusters
        if self.summary_clusters is not None:
            unique_clusters = set(self.summary_clusters)
            unique_clusters.discard(-1)  # Remove noise cluster

            for cluster_id in sorted(unique_clusters):
                cluster_texts = []
                for i, label in enumerate(self.summary_clusters):
                    if label == cluster_id:
                        cluster_texts.append(self.issue_summaries.iloc[i]["summary"])

                if cluster_texts:
                    all_clusters.append(
                        (f"IS-{cluster_id}", cluster_texts, "issue_summaries")
                    )

        print(
            f"{Colors.OKCYAN}📊 Total clusters to classify: {len(all_clusters)}{Colors.ENDC}"
        )

        # Process clusters in parallel
        cache_hits = 0
        api_calls = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_cluster = {
                executor.submit(
                    self.classify_cluster_optimized, cluster_data
                ): cluster_data[0]
                for cluster_data in all_clusters
            }

            # Process results as they complete
            for future in tqdm(
                as_completed(future_to_cluster),
                total=len(all_clusters),
                desc="Classifying clusters",
            ):
                cluster_id, theme, size, was_cached = future.result()

                if was_cached:
                    cache_hits += 1
                else:
                    api_calls += 1

                # Store result
                if cluster_id.startswith("DI-"):
                    actual_id = cluster_id[3:]  # Remove "DI-" prefix
                    classification_results["data_issues_clusters"][actual_id] = {
                        "theme": theme,
                        "size": size,
                    }
                else:  # IS-
                    actual_id = cluster_id[3:]  # Remove "IS-" prefix
                    classification_results["issue_summaries_clusters"][actual_id] = {
                        "theme": theme,
                        "size": size,
                    }

        # Print optimization summary
        print(f"\n{Colors.HEADER}{Colors.BOLD}⚡ OPTIMIZATION SUMMARY{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'=' * 50}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✅ Cache hits: {cache_hits} clusters{Colors.ENDC}")
        print(f"{Colors.OKCYAN}📡 API calls: {api_calls} clusters{Colors.ENDC}")
        print(
            f"{Colors.OKGREEN}💰 Cost savings: {cache_hits / len(all_clusters) * 100:.1f}%{Colors.ENDC}"
        )

        if cache_hits > 0:
            print(
                f"{Colors.OKGREEN}💾 Cache is working! Reusing previous classifications{Colors.ENDC}"
            )

        # Store classification results
        self.classification_results = classification_results

        # Extract themes for the new format
        data_issues_themes = {}
        issue_summaries_themes = {}

        for cluster_id, info in classification_results["data_issues_clusters"].items():
            data_issues_themes[cluster_id] = info["theme"]

        for cluster_id, info in classification_results[
            "issue_summaries_clusters"
        ].items():
            issue_summaries_themes[cluster_id] = info["theme"]

        # Print summary
        self.print_classification_summary(data_issues_themes, issue_summaries_themes)

        # Save classification results
        self.save_classification_results(data_issues_themes, issue_summaries_themes)

        return data_issues_themes, issue_summaries_themes

    def generate_excel_report(
        self,
        data_issues_df,
        issue_summaries_df,
        data_issues_clusters,
        issue_summaries_clusters,
        similar_clusters,
        unique_data_issues_clusters,
        unique_issue_summaries_clusters,
        data_issues_themes,
        issue_summaries_themes,
        similarity_threshold,
    ):
        """Generate a comprehensive Excel report with multiple tabs."""
        print(
            f"{Colors.OKCYAN}📊 Generating comprehensive Excel report...{Colors.ENDC}"
        )

        # Create Excel writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"clustering_analysis_report_{timestamp}.xlsx"

        with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:

            # Tab 1: Data Issues with Clusters & Themes
            print(f"   📋 Creating Tab 1: Data Issues with Clusters & Themes")
            data_issues_with_clusters = data_issues_df.copy()
            data_issues_with_clusters["Cluster"] = data_issues_clusters
            data_issues_with_clusters["Cluster_Theme"] = data_issues_with_clusters[
                "Cluster"
            ].map(
                lambda x: (
                    data_issues_themes.get(str(x), "Noise") if x != -1 else "Noise"
                )
            )
            data_issues_with_clusters.to_excel(
                writer, sheet_name="Data_Issues_Clustered", index=False
            )

            # Tab 2: Issue Summaries with Clusters & Themes
            print(f"   📋 Creating Tab 2: Issue Summaries with Clusters & Themes")
            issue_summaries_with_clusters = issue_summaries_df.copy()
            issue_summaries_with_clusters["Cluster"] = issue_summaries_clusters
            issue_summaries_with_clusters[
                "Cluster_Theme"
            ] = issue_summaries_with_clusters["Cluster"].map(
                lambda x: (
                    issue_summaries_themes.get(str(x), "Noise") if x != -1 else "Noise"
                )
            )
            issue_summaries_with_clusters.to_excel(
                writer, sheet_name="Issue_Summaries_Clustered", index=False
            )

            # Tab 3: Similar Clusters Analysis
            print(f"   📋 Creating Tab 3: Similar Clusters Analysis")

            # Collect all records from similar clusters
            similar_data_issues_records = []
            similar_issue_summaries_records = []

            for cluster_pair in similar_clusters:
                data_cluster_id, summary_cluster_id, similarity_score = cluster_pair

                # Get data issues for this cluster
                data_mask = data_issues_clusters == data_cluster_id
                data_issues_in_cluster = data_issues_df[data_mask].copy()
                data_issues_in_cluster["Cluster"] = data_cluster_id
                data_issues_in_cluster["Cluster_Theme"] = data_issues_themes.get(
                    str(data_cluster_id), "Unknown"
                )
                data_issues_in_cluster["Similarity_Score"] = similarity_score
                data_issues_in_cluster["Paired_Cluster_ID"] = summary_cluster_id
                similar_data_issues_records.append(data_issues_in_cluster)

                # Get issue summaries for this cluster
                summary_mask = issue_summaries_clusters == summary_cluster_id
                issue_summaries_in_cluster = issue_summaries_df[summary_mask].copy()
                issue_summaries_in_cluster["Cluster"] = summary_cluster_id
                issue_summaries_in_cluster["Cluster_Theme"] = (
                    issue_summaries_themes.get(str(summary_cluster_id), "Unknown")
                )
                issue_summaries_in_cluster["Similarity_Score"] = similarity_score
                issue_summaries_in_cluster["Paired_Cluster_ID"] = data_cluster_id
                similar_issue_summaries_records.append(issue_summaries_in_cluster)

            # Combine all similar cluster records
            if similar_data_issues_records:
                similar_data_issues_df = pd.concat(
                    similar_data_issues_records, ignore_index=True
                )
                similar_data_issues_df.to_excel(
                    writer, sheet_name="Similar_Clusters_Data_Issues", index=False
                )

            if similar_issue_summaries_records:
                similar_issue_summaries_df = pd.concat(
                    similar_issue_summaries_records, ignore_index=True
                )
                similar_issue_summaries_df.to_excel(
                    writer, sheet_name="Similar_Clusters_Issue_Summaries", index=False
                )

            # Tab 4: Unique Clusters Analysis
            print(f"   📋 Creating Tab 4: Unique Clusters Analysis")

            # Collect all records from unique clusters
            unique_data_issues_records = []
            unique_issue_summaries_records = []

            # Unique data issues clusters
            for cluster_id in unique_data_issues_clusters:
                data_mask = data_issues_clusters == cluster_id
                data_issues_in_cluster = data_issues_df[data_mask].copy()
                data_issues_in_cluster["Cluster"] = cluster_id
                data_issues_in_cluster["Cluster_Theme"] = data_issues_themes.get(
                    str(cluster_id), "Unknown"
                )
                data_issues_in_cluster["Cluster_Type"] = "Unique"
                unique_data_issues_records.append(data_issues_in_cluster)

            # Unique issue summaries clusters
            for cluster_id in unique_issue_summaries_clusters:
                summary_mask = issue_summaries_clusters == cluster_id
                issue_summaries_in_cluster = issue_summaries_df[summary_mask].copy()
                issue_summaries_in_cluster["Cluster"] = cluster_id
                issue_summaries_in_cluster["Cluster_Theme"] = (
                    issue_summaries_themes.get(str(cluster_id), "Unknown")
                )
                issue_summaries_in_cluster["Cluster_Type"] = "Unique"
                unique_issue_summaries_records.append(issue_summaries_in_cluster)

            # Combine all unique cluster records
            if unique_data_issues_records:
                unique_data_issues_df = pd.concat(
                    unique_data_issues_records, ignore_index=True
                )
                unique_data_issues_df.to_excel(
                    writer, sheet_name="Unique_Clusters_Data_Issues", index=False
                )

            if unique_issue_summaries_records:
                unique_issue_summaries_df = pd.concat(
                    unique_issue_summaries_records, ignore_index=True
                )
                unique_issue_summaries_df.to_excel(
                    writer, sheet_name="Unique_Clusters_Issue_Summaries", index=False
                )

            # Tab 5: Summary Statistics
            print(f"   📋 Creating Tab 5: Summary Statistics")
            summary_stats = []

            # Clustering metrics
            data_noise_count = np.sum(data_issues_clusters == -1)
            summary_noise_count = np.sum(issue_summaries_clusters == -1)
            data_cluster_count = len(set(data_issues_clusters)) - (
                1 if -1 in data_issues_clusters else 0
            )
            summary_cluster_count = len(set(issue_summaries_clusters)) - (
                1 if -1 in issue_summaries_clusters else 0
            )

            summary_stats.extend(
                [
                    {"Metric": "Total_Data_Issues", "Value": len(data_issues_df)},
                    {
                        "Metric": "Total_Issue_Summaries",
                        "Value": len(issue_summaries_df),
                    },
                    {"Metric": "Data_Issues_Clusters", "Value": data_cluster_count},
                    {
                        "Metric": "Issue_Summaries_Clusters",
                        "Value": summary_cluster_count,
                    },
                    {"Metric": "Data_Issues_Noise_Count", "Value": data_noise_count},
                    {
                        "Metric": "Issue_Summaries_Noise_Count",
                        "Value": summary_noise_count,
                    },
                    {
                        "Metric": "Similar_Clusters_Count",
                        "Value": len(similar_clusters),
                    },
                    {
                        "Metric": "Unique_Data_Clusters_Count",
                        "Value": len(unique_data_issues_clusters),
                    },
                    {
                        "Metric": "Unique_Summary_Clusters_Count",
                        "Value": len(unique_issue_summaries_clusters),
                    },
                    {"Metric": "Similarity_Threshold", "Value": similarity_threshold},
                    {
                        "Metric": "Analysis_Date",
                        "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                ]
            )

            # Temporal analysis (for data_issues)
            if "date" in data_issues_df.columns:
                data_issues_df["date"] = pd.to_datetime(data_issues_df["date"])
                date_range = f"{data_issues_df['date'].min().strftime('%Y-%m-%d')} to {data_issues_df['date'].max().strftime('%Y-%m-%d')}"
                summary_stats.append(
                    {"Metric": "Data_Issues_Date_Range", "Value": date_range}
                )

                # Monthly distribution
                monthly_counts = data_issues_df.groupby(
                    data_issues_df["date"].dt.to_period("M")
                ).size()
                for month, count in monthly_counts.items():
                    summary_stats.append(
                        {"Metric": f"Data_Issues_{month}", "Value": count}
                    )

            summary_stats_df = pd.DataFrame(summary_stats)
            summary_stats_df.to_excel(
                writer, sheet_name="Summary_Statistics", index=False
            )

            # Tab 6: Temporal Analysis (if dates available)
            if "date" in data_issues_df.columns:
                print(f"   📋 Creating Tab 6: Temporal Analysis")
                temporal_data = []

                # Add cluster themes to data issues for temporal analysis
                data_issues_with_themes = data_issues_df.copy()
                data_issues_with_themes["Cluster"] = data_issues_clusters
                data_issues_with_themes["Cluster_Theme"] = data_issues_with_themes[
                    "Cluster"
                ].map(
                    lambda x: (
                        data_issues_themes.get(str(x), "Noise") if x != -1 else "Noise"
                    )
                )
                data_issues_with_themes["date"] = pd.to_datetime(
                    data_issues_with_themes["date"]
                )
                data_issues_with_themes["Month"] = data_issues_with_themes[
                    "date"
                ].dt.to_period("M")
                data_issues_with_themes["Day_of_Week"] = data_issues_with_themes[
                    "date"
                ].dt.day_name()

                # Monthly theme distribution
                monthly_themes = (
                    data_issues_with_themes.groupby(["Month", "Cluster_Theme"])
                    .size()
                    .reset_index(name="Count")
                )
                monthly_themes["Month"] = monthly_themes["Month"].astype(str)
                monthly_themes.to_excel(
                    writer, sheet_name="Temporal_Analysis", index=False
                )

            # Tab 7: Cluster Details
            print(f"   📋 Creating Tab 7: Cluster Details")
            cluster_details = []

            # Data issues cluster details
            for cluster_id in set(data_issues_clusters):
                if cluster_id != -1:  # Skip noise
                    cluster_mask = data_issues_clusters == cluster_id
                    cluster_data = data_issues_df[cluster_mask]
                    cluster_details.append(
                        {
                            "File_Source": "Data_Issues",
                            "Cluster_ID": cluster_id,
                            "Cluster_Theme": data_issues_themes.get(
                                str(cluster_id), "Unknown"
                            ),
                            "Cluster_Size": len(cluster_data),
                            "Date_Range": (
                                f"{cluster_data['date'].min()} to {cluster_data['date'].max()}"
                                if "date" in cluster_data.columns
                                else "N/A"
                            ),
                            "Representative_Text": (
                                cluster_data["description"].iloc[0]
                                if len(cluster_data) > 0
                                else "N/A"
                            ),
                        }
                    )

            # Issue summaries cluster details
            for cluster_id in set(issue_summaries_clusters):
                if cluster_id != -1:  # Skip noise
                    cluster_mask = issue_summaries_clusters == cluster_id
                    cluster_data = issue_summaries_df[cluster_mask]
                    cluster_details.append(
                        {
                            "File_Source": "Issue_Summaries",
                            "Cluster_ID": cluster_id,
                            "Cluster_Theme": issue_summaries_themes.get(
                                str(cluster_id), "Unknown"
                            ),
                            "Cluster_Size": len(cluster_data),
                            "Date_Range": "N/A",
                            "Representative_Text": (
                                cluster_data["summary"].iloc[0]
                                if len(cluster_data) > 0
                                else "N/A"
                            ),
                        }
                    )

            cluster_details_df = pd.DataFrame(cluster_details)
            cluster_details_df.to_excel(
                writer, sheet_name="Cluster_Details", index=False
            )

        print(
            f"{Colors.OKGREEN}✅ Excel report saved as: {excel_filename}{Colors.ENDC}"
        )
        print(f"{Colors.OKCYAN}📊 Report contains 9 tabs:{Colors.ENDC}")
        print(f"   📋 Tab 1: Data Issues with Clusters & Themes")
        print(f"   📋 Tab 2: Issue Summaries with Clusters & Themes")
        print(f"   📋 Tab 3: Similar Clusters - Data Issues Records")
        print(f"   📋 Tab 4: Similar Clusters - Issue Summaries Records")
        print(f"   📋 Tab 5: Unique Clusters - Data Issues Records")
        print(f"   📋 Tab 6: Unique Clusters - Issue Summaries Records")
        print(f"   📋 Tab 7: Summary Statistics")
        print(f"   📋 Tab 8: Temporal Analysis (if dates available)")
        print(f"   📋 Tab 9: Cluster Details")

        return excel_filename

    def run_complete_analysis(
        self, similarity_threshold=0.3, optimized_classify=True, max_workers=4
    ):
        """Run the complete analysis pipeline."""
        print(f"{Colors.HEADER}🚀 Starting Complete ML Analysis Pipeline{Colors.ENDC}")
        print(f"{Colors.OKCYAN}{'=' * 60}{Colors.ENDC}")

        # Load data
        self.load_data()

        # Setup vector database
        self.setup_vector_database()

        # Run clustering analysis
        print(f"{Colors.OKCYAN}🔍 Running clustering analysis...{Colors.ENDC}")
        (
            data_issues_df,
            issue_summaries_df,
            data_issues_clusters,
            issue_summaries_clusters,
        ) = self.run_clustering_analysis()

        # Run cluster comparison
        print(f"{Colors.OKCYAN}🔍 Running cluster comparison...{Colors.ENDC}")
        (
            similar_clusters,
            unique_data_issues_clusters,
            unique_issue_summaries_clusters,
        ) = self.compare_clusters(
            data_issues_df,
            issue_summaries_df,
            data_issues_clusters,
            issue_summaries_clusters,
            similarity_threshold,
        )

        # Run binary classification for individual records
        print(f"{Colors.OKCYAN}🔍 Running binary data quality classification...{Colors.ENDC}")
        data_issues_df, issue_summaries_df = self.classify_all_records_binary(
            data_issues_df, issue_summaries_df
        )

        # Run thematic classification for clusters
        print(f"{Colors.OKCYAN}🔍 Running zero-shot thematic classification...{Colors.ENDC}")
        if optimized_classify:
            data_issues_themes, issue_summaries_themes = (
                self.classify_all_clusters_optimized(max_workers)
            )
        else:
            data_issues_themes, issue_summaries_themes = self.classify_all_clusters()

        # Print results
        self.print_comparison_summary(
            similar_clusters,
            unique_data_issues_clusters,
            unique_issue_summaries_clusters,
        )
        self.print_classification_summary(data_issues_themes, issue_summaries_themes)

        # Save results
        self.save_classification_results(data_issues_themes, issue_summaries_themes)

        # Generate Excel report
        excel_filename = self.generate_excel_report(
            data_issues_df,
            issue_summaries_df,
            data_issues_clusters,
            issue_summaries_clusters,
            similar_clusters,
            unique_data_issues_clusters,
            unique_issue_summaries_clusters,
            data_issues_themes,
            issue_summaries_themes,
            similarity_threshold,
        )

        # Setup RAG chat with all the analyzed data
        self.setup_rag_chat(
            data_issues_df,
            issue_summaries_df,
            data_issues_clusters,
            issue_summaries_clusters,
            data_issues_themes,
            issue_summaries_themes,
        )

        print(f"{Colors.OKCYAN}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}🎉 Complete analysis finished!{Colors.ENDC}")
        print(f"{Colors.OKCYAN}📊 Excel report: {excel_filename}{Colors.ENDC}")
        print(
            f"{Colors.OKCYAN}📄 JSON results: cluster_classifications.json{Colors.ENDC}"
        )
        print(f"{Colors.OKCYAN}📄 CSV themes: cluster_themes.csv{Colors.ENDC}")
        print(f"{Colors.OKCYAN}🤖 RAG Chat: Ready for interactive data exploration{Colors.ENDC}")

        return excel_filename

    def setup_rag_chat(self, data_issues_df, issue_summaries_df, data_issues_clusters, 
                       issue_summaries_clusters, data_issues_themes, issue_summaries_themes):
        """Setup RAG chat with all available data sources."""
        self.chat_data = {
            'data_issues_df': data_issues_df,
            'issue_summaries_df': issue_summaries_df,
            'data_issues_clusters': data_issues_clusters,
            'issue_summaries_clusters': issue_summaries_clusters,
            'data_issues_themes': data_issues_themes,
            'issue_summaries_themes': issue_summaries_themes
        }
        
        # Prepare searchable data
        self.prepare_searchable_data()
        
    def prepare_searchable_data(self):
        """Prepare data for semantic search and retrieval."""
        print(f"🔍 Preparing searchable data for RAG chat...")
        
        # Combine all text data for search
        self.searchable_texts = []
        self.text_metadata = []
        
        # Add data issues
        for idx, row in self.chat_data['data_issues_df'].iterrows():
            text = f"Data Issue {row['id']}: {row['description']}"
            metadata = {
                'source': 'data_issues',
                'id': row['id'],
                'date': row.get('date', ''),
                'cluster_id': self.chat_data['data_issues_clusters'][idx],
                'theme': self.chat_data['data_issues_themes'].get(str(self.chat_data['data_issues_clusters'][idx]), 'Unknown'),
                'data_quality_related': row.get('data_quality_related', 'Unknown'),
                'original_text': row['description']
            }
            self.searchable_texts.append(text)
            self.text_metadata.append(metadata)
        
        # Add issue summaries
        for idx, row in self.chat_data['issue_summaries_df'].iterrows():
            text = f"Issue Summary {row['Id_num']}: {row['summary']}"
            metadata = {
                'source': 'issue_summaries',
                'id': row['Id_num'],
                'cluster_id': self.chat_data['issue_summaries_clusters'][idx],
                'theme': self.chat_data['issue_summaries_themes'].get(str(self.chat_data['issue_summaries_clusters'][idx]), 'Unknown'),
                'data_quality_related': row.get('data_quality_related', 'Unknown'),
                'original_text': row['summary']
            }
            self.searchable_texts.append(text)
            self.text_metadata.append(metadata)
        
        print(f"   Prepared {len(self.searchable_texts)} searchable texts")
        
        # Store the data in ChromaDB for retrieval
        self._store_searchable_data_in_chromadb()
    
    def _store_searchable_data_in_chromadb(self):
        """Store searchable data in ChromaDB collections."""
        try:
            print(f"🔍 Storing searchable data in ChromaDB...")
            
            # Clear existing collections
            self.data_issues_collection.delete(where={})
            self.issue_summaries_collection.delete(where={})
            
            # Separate data issues and issue summaries
            data_issues_texts = []
            data_issues_metadatas = []
            issue_summaries_texts = []
            issue_summaries_metadatas = []
            
            for i, (text, metadata) in enumerate(zip(self.searchable_texts, self.text_metadata)):
                if metadata['source'] == 'data_issues':
                    data_issues_texts.append(text)
                    data_issues_metadatas.append(metadata)
                else:
                    issue_summaries_texts.append(text)
                    issue_summaries_metadatas.append(metadata)
            
            # Generate embeddings and store in collections
            if data_issues_texts:
                print(f"   Storing {len(data_issues_texts)} data issues...")
                embeddings = self.generate_titan_embeddings(data_issues_texts)
                ids = [f"data_issue_{i}" for i in range(len(data_issues_texts))]
                self.data_issues_collection.add(
                    embeddings=embeddings.tolist(),
                    documents=data_issues_texts,
                    metadatas=data_issues_metadatas,
                    ids=ids
                )
            
            if issue_summaries_texts:
                print(f"   Storing {len(issue_summaries_texts)} issue summaries...")
                embeddings = self.generate_titan_embeddings(issue_summaries_texts)
                ids = [f"issue_summary_{i}" for i in range(len(issue_summaries_texts))]
                self.issue_summaries_collection.add(
                    embeddings=embeddings.tolist(),
                    documents=issue_summaries_texts,
                    metadatas=issue_summaries_metadatas,
                    ids=ids
                )
            
            print(f"✅ Successfully stored {len(self.searchable_texts)} texts in ChromaDB")
            
        except Exception as e:
            print(f"❌ Error storing data in ChromaDB: {e}")
            # Fallback: use in-memory search
            print(f"⚠️ Falling back to in-memory search")
    
    def retrieve_relevant_data(self, query, top_k=5):
        """Retrieve relevant data using semantic search."""
        try:
            # Try ChromaDB first
            chromadb_results = self._retrieve_from_chromadb(query, top_k)
            if chromadb_results:
                return chromadb_results
            
            # Fallback to in-memory search if ChromaDB is empty
            return self._retrieve_from_memory(query, top_k)
            
        except Exception as e:
            print(f"{Colors.WARNING}⚠️ Error in retrieval: {e}{Colors.ENDC}")
            # Fallback to in-memory search
            return self._retrieve_from_memory(query, top_k)
    
    def _retrieve_from_chromadb(self, query, top_k=5):
        """Retrieve data from ChromaDB."""
        try:
            # Generate query embedding
            query_embeddings = self.generate_titan_embeddings([query])
            query_embedding = query_embeddings[0] if len(query_embeddings) > 0 else None
            
            if query_embedding is None:
                return []
            
            # Get all cached embeddings and documents
            all_embeddings = []
            all_documents = []
            all_metadatas = []
            
            for i, collection in enumerate([self.data_issues_collection, self.issue_summaries_collection]):
                collection_name = "data_issues" if i == 0 else "issue_summaries"
                
                if collection.count() > 0:
                    results = collection.get(include=['embeddings', 'documents', 'metadatas'])
                    
                    embeddings_data = results.get('embeddings')
                    documents_data = results.get('documents')
                    metadatas_data = results.get('metadatas')
                    
                    if embeddings_data is not None and documents_data is not None:
                        embeddings_len = len(embeddings_data)
                        documents_len = len(documents_data)
                        
                        if embeddings_len > 0 and documents_len > 0:
                            all_embeddings.extend(embeddings_data)
                            all_documents.extend(documents_data)
                            all_metadatas.extend(metadatas_data)
            
            if not all_embeddings:
                return []
            
            # Calculate similarities
            similarities = []
            
            for i, embedding in enumerate(all_embeddings):
                try:
                    if embedding is None:
                        continue
                    
                    # Convert to list if it's a numpy array to avoid truthiness issues
                    if hasattr(embedding, 'tolist'):
                        embedding_list = embedding.tolist()
                    else:
                        embedding_list = list(embedding)
                    
                    if len(embedding_list) == 0:
                        continue
                    
                    # Convert to numpy arrays for proper calculation
                    query_vec = np.array(query_embedding, dtype=float)
                    embed_vec = np.array(embedding_list, dtype=float)
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(query_vec, embed_vec)
                    query_norm = np.linalg.norm(query_vec)
                    embed_norm = np.linalg.norm(embed_vec)
                    
                    if query_norm > 0 and embed_norm > 0:
                        similarity = float(dot_product / (query_norm * embed_norm))
                        similarities.append((similarity, i))
                except Exception as e:
                    print(f"   ⚠️  Error calculating similarity for embedding {i}: {e}")
                    continue
            
            # Sort by similarity and return top results
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:top_k]]
            
            # Map back to documents and metadata
            relevant_data = []
            for idx in top_indices:
                if idx < len(all_documents):
                    relevant_data.append({
                        'text': all_documents[idx],
                        'metadata': all_metadatas[idx] if idx < len(all_metadatas) else {}
                    })
            
            print(f"   Returning {len(relevant_data)} relevant results from ChromaDB")
            return relevant_data
            
        except Exception as e:
            print(f"   ⚠️  Error retrieving from ChromaDB: {e}")
            return []
    
    def _retrieve_from_memory(self, query, top_k=5):
        """Retrieve data from in-memory searchable texts."""
        try:
            if not hasattr(self, 'searchable_texts') or not self.searchable_texts:
                print("   ⚠️  No searchable texts available in memory")
                return []
            
            print(f"   Searching {len(self.searchable_texts)} texts in memory...")
            
            # Simple keyword-based search as fallback
            query_lower = query.lower()
            relevant_data = []
            
            for i, (text, metadata) in enumerate(zip(self.searchable_texts, self.text_metadata)):
                text_lower = text.lower()
                # Simple relevance score based on keyword matches
                relevance_score = 0
                for word in query_lower.split():
                    if word in text_lower:
                        relevance_score += 1
                
                if relevance_score > 0:
                    relevant_data.append({
                        'text': text,
                        'metadata': metadata,
                        'relevance_score': relevance_score
                    })
            
            # Sort by relevance score and return top results
            relevant_data.sort(key=lambda x: x['relevance_score'], reverse=True)
            relevant_data = relevant_data[:top_k]
            
            # Remove relevance_score from metadata for consistency
            for item in relevant_data:
                if 'relevance_score' in item:
                    del item['relevance_score']
            
            print(f"   Returning {len(relevant_data)} relevant results from memory")
            return relevant_data
            
        except Exception as e:
            print(f"   ⚠️  Error in memory search: {e}")
            return []
    
    def generate_chat_response(self, query, context_data, chat_history):
        """Generate response using Llama model with context."""
        try:
            # Prepare context from retrieved data
            context_text = ""
            if context_data and isinstance(context_data, list):
                context_text = "Relevant data:\n"
                for i, item in enumerate(context_data[:3]):  # Use top 3 results
                    if isinstance(item, dict) and 'text' in item:
                        context_text += f"{i+1}. {item['text']}\n"
                        
                        # Safely access metadata fields
                        metadata = item.get('metadata', {})
                        if isinstance(metadata, dict):
                            theme = metadata.get('theme', 'Unknown')
                            data_quality = metadata.get('data_quality_related', 'Unknown')
                            context_text += f"   Theme: {theme}\n"
                            context_text += f"   Data Quality: {data_quality}\n\n"
                        else:
                            context_text += f"   Metadata: {metadata}\n\n"
            
            # Prepare conversation history
            history_text = ""
            if chat_history.conversation:
                history_text = "Previous conversation:\n"
                for turn in chat_history.conversation[-3:]:  # Last 3 turns
                    history_text += f"User: {turn['user_query']}\n"
                    history_text += f"Assistant: {turn['assistant_response'][:200]}...\n\n"
            
            # Create prompt
            prompt = f"""You are a helpful data analyst assistant. You have access to company issue data with clustering analysis, themes, and binary classifications.

{history_text}
{context_text}

Current user question: {query}

Please provide a helpful, informative response based on the available data. If the question is about specific records, mention the IDs. If it's about trends or patterns, provide insights. Be conversational but professional.

Response:"""

            # Use the correct Llama 3 instruction format as per AWS documentation
            formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

            # Prepare request
            native_request = {
                "prompt": formatted_prompt,
                "max_gen_len": 512,
                "temperature": 0.5,
                
            }
            request = json.dumps(native_request)

            # Call Llama model with streaming
            response_stream = self.bedrock_client.invoke_model_with_response_stream(
                modelId=self.llama_model_id, body=request
            )

            # Collect streaming response
            response_text = ""
            for event in response_stream["body"]:
                chunk = json.loads(event['chunk']["bytes"])
                if 'generation' in chunk:
                    chunk_text = chunk['generation']
                    response_text += chunk_text
                    print(chunk_text, end="", flush=True)

            
            if not response_text:
                response_text = "I'm sorry, I couldn't generate a response at this time."

            # Clean up response
            response_text = response_text.strip()
            
            return response_text

        except Exception as e:
            print(f"{Colors.FAIL}❌ Error generating chat response: {e}{Colors.ENDC}")
            return "I'm sorry, I encountered an error while processing your request."

    def generate_chat_response_streaming(self, query, context_data, chat_history):
        """Generate streaming response using Llama model with context."""
        try:
            
            # Prepare context from retrieved data
            context_text = ""
            if context_data and isinstance(context_data, list):
                context_text = "Relevant data from your analysis:\n"
                for i, item in enumerate(context_data[:3]):  # Use top 3 results
                    if isinstance(item, dict) and 'text' in item:
                        # Clean and truncate text for better readability
                        text = item['text'][:500] + "..." if len(item['text']) > 500 else item['text']
                        context_text += f"{i+1}. {text}\n"
                        
                        # Safely access metadata fields
                        metadata = item.get('metadata', {})
                        if isinstance(metadata, dict):
                            theme = metadata.get('theme', 'Unknown')
                            data_quality = metadata.get('data_quality_related', 'Unknown')
                            cluster_id = metadata.get('cluster_id', 'Unknown')
                            context_text += f"   Cluster: {cluster_id} | Theme: {theme} | Data Quality Issue: {data_quality}\n\n"
                        else:
                            context_text += f"   Metadata: {metadata}\n\n"
            
            # Prepare conversation history
            history_text = ""
            if chat_history.conversation:
                history_text = "Previous conversation:\n"
                for turn in chat_history.conversation[-3:]:  # Last 3 turns
                    history_text += f"User: {turn['user_query']}\n"
                    history_text += f"Assistant: {turn['assistant_response'][:200]}...\n\n"
            
            # Create a more specific, data-driven prompt
            if context_text:
                prompt = f"""You are analyzing company data quality issues. Here is the relevant data:

{context_text}

{history_text}

Question: {query}

Answer the question based ONLY on the data provided above. Be specific and factual. If the data shows numbers, mention them. If the data shows themes or patterns, describe them. Do not provide examples, code snippets, or generic explanations. Give a direct answer based on what the data tells us."""
            else:
                prompt = f"""You are analyzing company data quality issues, but no relevant data was found for this specific question.

{history_text}

Question: {query}

I don't have specific data to answer this question. I can help with questions about data quality issues, clustering results, themes, or specific records from the analysis. What would you like to know about the data we have?"""
            
            # Use the correct Llama 3 instruction format as per AWS documentation
            formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

            # Prepare request with more conservative parameters for factual responses
            native_request = {
                "prompt": formatted_prompt,
                "max_gen_len": 256,  # Shorter for more focused responses
                "temperature": 0.3,  # Lower temperature for more factual responses
                "top_p": 0.9,  # Keep top_p for response quality
            }
            request = json.dumps(native_request)

            # Call Llama model with streaming
            response_stream = self.bedrock_client.invoke_model_with_response_stream(
                modelId=self.llama_model_id, body=request
            )

            # Stream response in real-time
            response_text = ""
            
            # Process the streaming response according to AWS Bedrock documentation
            for event in response_stream["body"]:
                # Parse the chunk according to AWS Bedrock format
                chunk = json.loads(event["chunk"]["bytes"])
                
                if "generation" in chunk:
                    chunk_text = chunk["generation"]
                    response_text += chunk_text
                    print(chunk_text, end="", flush=True)
            
            if not response_text:
                response_text = "I'm sorry, I couldn't generate a response at this time."
                print(response_text, end="", flush=True)
            
            # Clean up response
            response_text = response_text.strip()
            
            return response_text

        except Exception as e:
            error_msg = f"I'm sorry, I encountered an error while processing your request: {e}"
            print(error_msg, end="", flush=True)
            return error_msg

    def start_rag_chat(self):
        """Start the RAG chat interface."""
        if not hasattr(self, 'chat_data'):
            print(f"{Colors.FAIL}❌ RAG chat not initialized. Please run complete analysis first.{Colors.ENDC}")
            return
        
        self.chat_with_data()

    def chat_with_data(self):
        """Interactive chat interface with full history and recollection."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}🤖 RAG CHAT INTERFACE{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'=' * 60}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Hello! I can help you explore your data. Ask me anything about:{Colors.ENDC}")
        print(f"   • Data quality issues and classifications")
        print(f"   • Cluster themes and relationships")
        print(f"   • Temporal patterns and trends")
        print(f"   • Similar and unique clusters")
        print(f"   • Any specific records or topics")
        print(f"\n{Colors.WARNING}Type 'quit', 'exit', or 'bye' to end the chat{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'=' * 60}{Colors.ENDC}")

        chat_history = ChatHistory()
        
        while True:
            try:
                user_input = input(f"\n{Colors.OKGREEN}You: {Colors.ENDC}")
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print(f"\n{Colors.HEADER}🤖 RAG Chat: {Colors.ENDC}Goodbye! {chat_history.summarize_session()}")
                    print(f"{Colors.OKCYAN}Your analysis results are saved in the Excel report.{Colors.ENDC}")
                    break
                
                # Check for clear history
                if user_input.lower() in ['clear', 'reset', 'new']:
                    chat_history = ChatHistory()
                    print(f"{Colors.HEADER}🤖 RAG Chat: {Colors.ENDC}Chat history cleared. Starting fresh conversation.")
                    continue
                
                if not user_input.strip():
                    continue
                
                # Get context from previous conversation
                context = chat_history.get_context_for_query(user_input)
                
                # Retrieve relevant data
                relevant_data = self.retrieve_relevant_data(user_input)
                
                # Generate and display streaming response
                print(f"{Colors.HEADER}🤖 RAG Chat: {Colors.ENDC}", end="", flush=True)
                response = self.generate_chat_response_streaming(user_input, relevant_data, chat_history)
                print()  # New line after streaming
                
                # Store in history
                chat_history.add_exchange(user_input, response, relevant_data)
                
            except KeyboardInterrupt:
                print(f"\n{Colors.HEADER}🤖 RAG Chat: {Colors.ENDC}Chat interrupted. {chat_history.summarize_session()}")
                break
            except Exception as e:
                print(f"{Colors.FAIL}❌ Error in chat: {e}{Colors.ENDC}")
                continue


class ChatHistory:
    """Manages conversation history and context."""
    
    def __init__(self):
        self.conversation = []
        self.context_summary = ""
        self.session_start = datetime.now()
    
    def add_exchange(self, user_query, assistant_response, relevant_data):
        """Add a conversation turn with context."""
        turn = {
            'timestamp': datetime.now(),
            'user_query': user_query,
            'assistant_response': assistant_response,
            'relevant_data': relevant_data
        }
        self.conversation.append(turn)
        
        # Update context summary
        self.update_context_summary()
    
    def get_context_for_query(self, current_query):
        """Get relevant context from previous conversation."""
        if not self.conversation:
            return ""
        
        # Look for references to previous topics
        context_parts = []
        
        # Check for pronouns and references
        if any(word in current_query.lower() for word in ['that', 'those', 'them', 'it', 'this']):
            if self.conversation:
                last_turn = self.conversation[-1]
                context_parts.append(f"Previous query: {last_turn['user_query']}")
                context_parts.append(f"Previous response: {last_turn['assistant_response'][:100]}...")
        
        # Check for temporal references
        if any(word in current_query.lower() for word in ['january', 'february', 'march', 'april', 'month']):
            context_parts.append("Temporal context: Data spans from January to April 2024")
        
        # Check for theme references
        if any(word in current_query.lower() for word in ['theme', 'cluster', 'group']):
            context_parts.append("Cluster context: Data is organized into thematic clusters")
        
        return " | ".join(context_parts)
    
    def update_context_summary(self):
        """Update the context summary based on conversation."""
        if len(self.conversation) <= 3:
            return
        
        # Create a summary of what's been discussed
        topics = set()
        for turn in self.conversation[-5:]:  # Last 5 turns
            query = turn['user_query'].lower()
            if 'data quality' in query:
                topics.add('data quality issues')
            if 'cluster' in query or 'theme' in query:
                topics.add('clustering analysis')
            if 'january' in query or 'february' in query or 'march' in query:
                topics.add('temporal analysis')
            if 'similar' in query:
                topics.add('cluster similarities')
        
        self.context_summary = f"Discussed topics: {', '.join(topics)}"
    
    def summarize_session(self):
        """Create a summary of the chat session."""
        if not self.conversation:
            return "No conversation to summarize."
        
        duration = datetime.now() - self.session_start
        total_exchanges = len(self.conversation)
        
        summary = f"Chat session summary:\n"
        summary += f"• Duration: {duration.total_seconds():.1f} seconds\n"
        summary += f"• Total exchanges: {total_exchanges}\n"
        summary += f"• Topics discussed: {self.context_summary}\n"
        
        return summary


def main():
    """Main function to run the complete analysis."""
    # Check command line arguments
    similarity_threshold = 0.3
    clear_cache = False
    rebuild_cache = False
    test_cache_only = False
    reset_db = False
    start_mlflow_ui = False
    mlflow_port = 5000
    classify_only = False
    optimized_classify = False
    max_workers = 4

    if len(sys.argv) > 1:
        if sys.argv[1] == "--mlflow-ui":
            start_mlflow_ui = True
            if len(sys.argv) > 2:
                try:
                    mlflow_port = int(sys.argv[2])
                except ValueError:
                    print(
                        f"{Colors.FAIL}❌ Invalid port number. Using default 5000{Colors.ENDC}"
                    )
                    mlflow_port = 5000
        elif sys.argv[1] == "--test-cache":
            test_cache_only = True
        elif sys.argv[1] == "--reset-db":
            reset_db = True
            if len(sys.argv) > 2:
                try:
                    similarity_threshold = float(sys.argv[2])
                    if not (0 <= similarity_threshold <= 1):
                        print(
                            f"{Colors.FAIL}❌ Similarity threshold must be between 0 and 1. Using default 0.3{Colors.ENDC}"
                        )
                        similarity_threshold = 0.3
                except ValueError:
                    print(
                        f"{Colors.FAIL}❌ Invalid similarity threshold. Using default 0.3{Colors.ENDC}"
                    )
                    similarity_threshold = 0.3
        elif sys.argv[1] == "--force-reset":
            # Force reset the database and exit
            print(f"{Colors.ORANGE}🔄 Force resetting vector database...{Colors.ENDC}")
            analysis = CompleteMLAnalysis()
            analysis.load_data()
            analysis.setup_vector_database()
            if analysis.reset_vector_database():
                print(f"{Colors.OKGREEN}✅ Database reset successful!{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}❌ Database reset failed!{Colors.ENDC}")
            return
        elif sys.argv[1] == "--clear-cache":
            clear_cache = True
            if len(sys.argv) > 2:
                try:
                    similarity_threshold = float(sys.argv[2])
                    if not (0 <= similarity_threshold <= 1):
                        print(
                            f"{Colors.FAIL}❌ Similarity threshold must be between 0 and 1. Using default 0.3{Colors.ENDC}"
                        )
                        similarity_threshold = 0.3
                except ValueError:
                    print(
                        f"{Colors.FAIL}❌ Invalid similarity threshold. Using default 0.3{Colors.ENDC}"
                    )
                    similarity_threshold = 0.3
        elif sys.argv[1] == "--rebuild-cache":
            rebuild_cache = True
            if len(sys.argv) > 2:
                try:
                    similarity_threshold = float(sys.argv[2])
                    if not (0 <= similarity_threshold <= 1):
                        print(
                            f"{Colors.FAIL}❌ Similarity threshold must be between 0 and 1. Using default 0.3{Colors.ENDC}"
                        )
                        similarity_threshold = 0.3
                except ValueError:
                    print(
                        f"{Colors.FAIL}❌ Invalid similarity threshold. Using default 0.3{Colors.ENDC}"
                    )
                    similarity_threshold = 0.3
        elif sys.argv[1] == "--classify-only":
            classify_only = True
            if len(sys.argv) > 2:
                try:
                    similarity_threshold = float(sys.argv[2])
                    if not (0 <= similarity_threshold <= 1):
                        print(
                            f"{Colors.FAIL}❌ Similarity threshold must be between 0 and 1. Using default 0.3{Colors.ENDC}"
                        )
                        similarity_threshold = 0.3
                except ValueError:
                    print(
                        f"{Colors.FAIL}❌ Invalid similarity threshold. Using default 0.3{Colors.ENDC}"
                    )
                    similarity_threshold = 0.3
        elif sys.argv[1] == "--optimized-classify":
            optimized_classify = True
        elif sys.argv[1] == "--binary-classify":
            # Test binary classification only
            print(f"{Colors.HEADER}🔍 Testing Binary Data Quality Classification{Colors.ENDC}")
            print(f"{Colors.OKCYAN}{'=' * 60}{Colors.ENDC}")
            
            analysis = CompleteMLAnalysis()
            analysis.load_data()
            
            # Load the data
            data_issues_df = pd.read_csv("data_issues.csv")
            issue_summaries_df = pd.read_csv("issue_summaries.csv")
            
            # Run binary classification
            data_issues_df, issue_summaries_df = analysis.classify_all_records_binary(
                data_issues_df, issue_summaries_df
            )
            
            # Save results
            data_issues_df.to_csv("data_issues_with_binary_classification.csv", index=False)
            issue_summaries_df.to_csv("issue_summaries_with_binary_classification.csv", index=False)
            
            print(f"\n{Colors.OKGREEN}✅ Binary classification completed!{Colors.ENDC}")
            print(f"{Colors.OKCYAN}📄 Results saved to:{Colors.ENDC}")
            print(f"   • data_issues_with_binary_classification.csv")
            print(f"   • issue_summaries_with_binary_classification.csv")
            return
        elif sys.argv[1] == "--rag-chat":
            # Start RAG chat interface
            print(f"{Colors.HEADER}🤖 Starting RAG Chat Interface{Colors.ENDC}")
            print(f"{Colors.OKCYAN}{'=' * 60}{Colors.ENDC}")
            
            analysis = CompleteMLAnalysis()
            
            # Check if we have existing analysis results
            if os.path.exists("clustering_analysis_report_latest.xlsx"):
                print(f"{Colors.OKCYAN}📊 Loading existing analysis results...{Colors.ENDC}")
                # Load data and setup chat
                analysis.load_data()
                analysis.setup_vector_database()
                
                # Load the data
                data_issues_df = pd.read_csv("data_issues.csv")
                issue_summaries_df = pd.read_csv("issue_summaries.csv")
                
                # Run clustering analysis to get clusters and themes
                (data_issues_df, issue_summaries_df, 
                 data_issues_clusters, issue_summaries_clusters) = analysis.run_clustering_analysis()
                
                # Run classification
                data_issues_themes, issue_summaries_themes = analysis.classify_all_clusters_optimized()
                
                # Setup RAG chat
                analysis.setup_rag_chat(
                    data_issues_df, issue_summaries_df, 
                    data_issues_clusters, issue_summaries_clusters,
                    data_issues_themes, issue_summaries_themes
                )
                
                # Start chat
                analysis.start_rag_chat()
            else:
                print(f"{Colors.WARNING}⚠️ No existing analysis found. Running complete analysis first...{Colors.ENDC}")
                analysis.run_complete_analysis()
                analysis.start_rag_chat()
            return
        elif sys.argv[1] == "--max-workers":
            if len(sys.argv) > 2:
                try:
                    max_workers = int(sys.argv[2])
                    if max_workers < 1:
                        print(
                            f"{Colors.FAIL}❌ Max workers must be at least 1. Using default 4{Colors.ENDC}"
                        )
                        max_workers = 4
                except ValueError:
                    print(
                        f"{Colors.FAIL}❌ Invalid max workers. Using default 4{Colors.ENDC}"
                    )
                    max_workers = 4
            if len(sys.argv) > 3:
                try:
                    similarity_threshold = float(sys.argv[3])
                    if not (0 <= similarity_threshold <= 1):
                        print(
                            f"{Colors.FAIL}❌ Similarity threshold must be between 0 and 1. Using default 0.3{Colors.ENDC}"
                        )
                        similarity_threshold = 0.3
                except ValueError:
                    print(
                        f"{Colors.FAIL}❌ Invalid similarity threshold. Using default 0.3{Colors.ENDC}"
                    )
                    similarity_threshold = 0.3
        elif sys.argv[1] == "--help":
            print(
                f"{Colors.HEADER}{Colors.BOLD}Complete ML Analysis - Usage Options:{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py [similarity_threshold]{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --mlflow-ui [port]{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --test-cache{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --reset-db [similarity_threshold]{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --clear-cache [similarity_threshold]{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --rebuild-cache [similarity_threshold]{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --classify-only [similarity_threshold]{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --optimized-classify [max_workers] [similarity_threshold]{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --binary-classify{Colors.ENDC}"
            )
            print(
                f"{Colors.OKBLUE}  python complete_ml_analysis.py --rag-chat{Colors.ENDC}"
            )
            print(
                f"{Colors.OKCYAN}  similarity_threshold: Float between 0 and 1 (default: 0.3){Colors.ENDC}"
            )
            print(
                f"{Colors.OKCYAN}  max_workers: Number of parallel workers (default: 4){Colors.ENDC}"
            )
            print(f"{Colors.OKCYAN}  port: MLflow UI port (default: 5000){Colors.ENDC}")
            print(f"{Colors.OKCYAN}  --binary-classify: Test Yes/No data quality classification only{Colors.ENDC}")
            print(f"{Colors.OKCYAN}  --rag-chat: Start interactive RAG chat with streaming responses{Colors.ENDC}")
            return
        else:
            try:
                similarity_threshold = float(sys.argv[1])
                if not (0 <= similarity_threshold <= 1):
                    print(
                        f"{Colors.FAIL}❌ Similarity threshold must be between 0 and 1. Using default 0.3{Colors.ENDC}"
                    )
                    similarity_threshold = 0.3
            except ValueError:
                print(
                    f"{Colors.FAIL}❌ Invalid similarity threshold. Using default 0.3{Colors.ENDC}"
                )
                similarity_threshold = 0.3

    # Run complete analysis
    analyzer = CompleteMLAnalysis()

    if start_mlflow_ui:
        print(
            f"{Colors.OKCYAN}{Colors.BOLD}🌐 Starting MLflow UI on port {mlflow_port}...{Colors.ENDC}"
        )
        analyzer.start_mlflow_ui(mlflow_port)
        return
    elif test_cache_only:
        print(f"{Colors.ORANGE}🧪 Testing cache functionality only...{Colors.ENDC}")
        analyzer.load_data()
        analyzer.setup_vector_database()
        print(f"{Colors.OKGREEN}✅ Cache test completed{Colors.ENDC}")
        return
    elif reset_db:
        print(
            f"{Colors.WARNING}⚠️  Resetting vector database to fix dimension issues...{Colors.ENDC}"
        )
        analyzer.load_data()
        analyzer.reset_vector_database()
        print(
            f"{Colors.OKGREEN}✅ Database reset completed. Running analysis...{Colors.ENDC}"
        )
    elif clear_cache:
        print(f"{Colors.WARNING}🗑️  Clearing cache before analysis...{Colors.ENDC}")
        analyzer.load_data()
        analyzer.setup_vector_database()
        analyzer.clear_cache()
    elif rebuild_cache:
        print(f"{Colors.WARNING}🔄 Rebuilding cache before analysis...{Colors.ENDC}")
        analyzer.load_data()
        analyzer.setup_vector_database()
        analyzer.rebuild_cache()
    elif classify_only:
        print(
            f"{Colors.OKCYAN}🎯 Running classification only (requires existing clustering results)...{Colors.ENDC}"
        )
        analyzer.load_data()
        analyzer.setup_vector_database()
        analyzer.run_clustering_analysis()  # Need clustering results first
        analyzer.classify_all_clusters()
        print(f"{Colors.OKGREEN}✅ Classification completed!{Colors.ENDC}")
        return
    elif optimized_classify:
        print(
            f"{Colors.OKCYAN}🚀 Running optimized classification only (requires existing clustering results)...{Colors.ENDC}"
        )
        print(
            f"{Colors.OKCYAN}⚡ Using {max_workers} parallel workers for faster processing{Colors.ENDC}"
        )
        analyzer.load_data()
        analyzer.setup_vector_database()
        analyzer.run_clustering_analysis()  # Need clustering results first
        analyzer.classify_all_clusters_optimized(max_workers=max_workers)
        print(f"{Colors.OKGREEN}✅ Optimized classification completed!{Colors.ENDC}")
        return

    analyzer.run_complete_analysis(similarity_threshold)


if __name__ == "__main__":
    main()
