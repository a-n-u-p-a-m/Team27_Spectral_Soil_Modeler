"""
Streamlit Web Application - Spectral Soil Modeler
==================================================
Interactive web interface with separate Training and Prediction modes.

Features:
- THREE training options (Standard, Tuned Only, Both Parallel)
- Separate UI views for Training vs Prediction
- Comprehensive error handling
- Real-time visualizations
- Model persistence and inference
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import logging
import os
from datetime import datetime
import joblib
from typing import Dict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file FIRST (before any other imports that need them)
import os
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"\'')

# --- CORE ML IMPORTS (KEPT SEPARATE) ---
from data_loader import DataLoader
from preprocessing import SpectralPreprocessor
from models.model_trainer import ModelTrainer
from evaluation import ModelEvaluator

# --- CONSOLIDATED APP IMPORTS ---
from system import initialize_logging, get_logger, get_performance_tracker, ModelPersistence
from interface import (
    apply_modern_style,
    render_training_results_section,
    render_prediction_model_inspection,
    render_comparison_mode,
    DataProfiler
)
from services import ChatInterface

# Mark enhancements as available (now consolidated into interface and services)
ENHANCEMENTS_AVAILABLE = True

# Configure logging
logger = initialize_logging(log_dir="./logs")
perf_tracker = get_performance_tracker()

# Streamlit configuration
st.set_page_config(
    page_title="Spectral Soil Modeler",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Glassmorphism styling IMMEDIATELY (before any other content)
from interface import apply_modern_style
apply_modern_style()


def init_session_state():
    """Initialize session state variables."""
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = 'training'  # 'training' or 'prediction'
    
    # Training mode states
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'trainer_standard' not in st.session_state:
        st.session_state.trainer_standard = None
    if 'trainer_tuned' not in st.session_state:
        st.session_state.trainer_tuned = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    if 'evaluation_report' not in st.session_state:
        st.session_state.evaluation_report = None
    if 'all_training_results' not in st.session_state:
        st.session_state.all_training_results = {}
    
    # Prediction mode states
    if 'loaded_model' not in st.session_state:
        st.session_state.loaded_model = None
    if 'loaded_preprocessor' not in st.session_state:
        st.session_state.loaded_preprocessor = None
    if 'model_metadata' not in st.session_state:
        st.session_state.model_metadata = None
    
    # UI state flags for disabling during operations
    if 'report_generating' not in st.session_state:
        st.session_state.report_generating = False
    if 'ai_thinking' not in st.session_state:
        st.session_state.ai_thinking = False
    
    # AI provider selection
    if 'ai_provider' not in st.session_state:
        st.session_state.ai_provider = 'gemini'  # Default to Gemini


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def display_basic_results(all_results: Dict):
    """Fallback display for results when enhancements are not available."""
    for config_name, data in all_results.items():
        st.markdown(f'<div class="section-header">📊 Results: {config_name} Training</div>', unsafe_allow_html=True)
        
        results_df = data['results']
        best_result, best_model = data['best_model']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best R²", f"{results_df['Test_R²'].max():.4f}")
        with col2:
            st.metric("Mean R²", f"{results_df['Test_R²'].mean():.4f}")
        with col3:
            st.metric("Best RMSE", f"{results_df['Test_RMSE'].min():.4f}")
        with col4:
            st.metric("Mean RMSE", f"{results_df['Test_RMSE'].mean():.4f}")
        
        # Leaderboard
        st.subheader("🏆 Top 10 Models")
        leaderboard = results_df.nlargest(10, 'Test_R²')[['Model', 'Technique', 'Test_R²', 'Test_RMSE', 'RPD']]
        st.dataframe(leaderboard, width='stretch')
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                results_df.groupby('Model')['Test_R²'].mean().sort_values(),
                labels={'value': 'Mean Test R²'},
                title=f'{config_name}: Average Test R² by Model'
            )
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            fig2 = px.bar(
                results_df.groupby('Technique')['Test_R²'].mean().sort_values(),
                labels={'value': 'Mean Test R²'},
                title=f'{config_name}: Average Test R² by Technique'
            )
            st.plotly_chart(fig2, width='stretch')
        
        # Best model info
        st.markdown(f"**✨ Best Model ({config_name}):**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Algorithm", best_result['Model'])
        with col2:
            st.metric("Technique", best_result['Technique'])
        with col3:
            st.metric("Test R²", f"{best_result['Test_R²']:.4f}")
        with col4:
            st.metric("RPD", f"{best_result['RPD']:.2f}")


# ============================================================================
# TRAINING MODE FUNCTIONS
# ============================================================================

# ============================================================================
# TRAINING MODE FUNCTIONS
# ============================================================================

def training_mode():
    """Training mode interface."""
    st.markdown(
        '<div class="main-header">🌾 Spectral Soil Modeler - TRAINING MODE</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("""
    **Train and compare 15 model-technique combinations**  
    Choose from 3 training paradigms: Standard, Tuned Only, or Both (Parallel).
    """)
    
    # Step 1: Load Data
    st.markdown('<div class="section-header">📁 Step 1: Load Data</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload spectral data (XLS/CSV)",
        type=['xlsx', 'csv', 'xls'],
        help="File should contain spectral features and a target column",
        key="training_upload_data",
        disabled=st.session_state.get('training_in_progress', False) or st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
    )
    
    # If training already complete and data is loaded, skip file upload check
    if not uploaded_file and not st.session_state.training_complete:
        st.info("👆 Please upload a spectral data file to begin")
        return
    
    # Load and store data (only if file uploaded and not already loaded)
    if uploaded_file and not st.session_state.data_loaded:
        try:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            data_loader = DataLoader()
            raw_data = data_loader.load_data(temp_path)
            
            st.session_state.data_loaded = True
            st.session_state.data_loader = data_loader
            st.session_state.raw_data = raw_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", raw_data.shape[0])
            with col2:
                st.metric("Features", raw_data.shape[1])
            with col3:
                st.metric("Memory", f"{raw_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            with st.expander("View Data Sample"):
                st.dataframe(raw_data.head(10), width='stretch')
        
        except Exception as e:
            st.markdown(f'<div class="error-box"><b>❌ Error loading data:</b><br>{str(e)}</div>', unsafe_allow_html=True)
            logger.error(f"Error loading data: {str(e)}", component="app")
            return
    
    # Use existing data if already loaded
    if st.session_state.data_loaded and st.session_state.raw_data is not None:
        raw_data = st.session_state.raw_data
        
        # Show data info even on reruns
        if not st.session_state.training_complete:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", raw_data.shape[0])
            with col2:
                st.metric("Features", raw_data.shape[1])
            with col3:
                st.metric("Memory", f"{raw_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
    else:
        return
    
    # ============================================================================
    # IF TRAINING ALREADY COMPLETE, SKIP TO RESULTS DISPLAY
    # ============================================================================
    if st.session_state.training_complete:
        if 'all_training_results' in st.session_state and st.session_state.all_training_results:
            all_results = st.session_state.all_training_results
            
            # Determine if we're in "Both" mode (multiple paradigms)
            is_both_mode = len(all_results) > 1
            
            # Display results using enhanced UI if available
            if ENHANCEMENTS_AVAILABLE:
                try:
                    # Display results for each training type with enhanced features
                    # When in "Both" mode, skip AI tab for individual paradigms (will show in comparison)
                    for config_name, data in all_results.items():
                        results_df = data['results']
                        try:
                            render_training_results_section(
                                results_df=results_df,
                                paradigm=config_name,
                                include_exports=True,
                                include_reports=True,
                                include_analytics=True,
                                is_individual_paradigm=not is_both_mode
                            )
                            st.markdown("---")
                        except Exception as tab_error:
                            logger.warning(f"Error rendering {config_name} results tab: {str(tab_error)}")
                            st.warning(f"Could not render enhanced view for {config_name}: {str(tab_error)}")
                            # Show basic results for this paradigm
                            display_basic_results({config_name: data})
                    
                    # Comparison view if both standard and tuned available
                    if 'Standard' in all_results and 'Tuned' in all_results:
                        try:
                            st.markdown("## 🔄 Training Paradigm Comparison")
                            render_comparison_mode(
                                all_results['Standard']['results'],
                                all_results['Tuned']['results']
                            )
                        except Exception as comp_error:
                            logger.warning(f"Error rendering comparison: {str(comp_error)}")
                            st.info("Comparison view not available")
                
                except Exception as e:
                    logger.warning(f"Error in enhanced results display: {str(e)}")
                    st.info("Using basic results display...")
                    # Fallback to basic display
                    display_basic_results(all_results)
            else:
                # Fallback to basic display if enhancements not available
                display_basic_results(all_results)
        
        return  # Exit after displaying results
    
    # ============================================================================
    # TRAINING SETUP SECTION - ONLY SHOWN IF TRAINING NOT YET COMPLETE
    # ============================================================================
    
    # Step 2: Select Target
    st.markdown('<div class="section-header">🎯 Step 2: Select Target Column</div>', unsafe_allow_html=True)
    
    target_col = st.selectbox(
        "Select target column (soil property to predict)",
        raw_data.columns.tolist(),
        disabled=st.session_state.get('training_in_progress', False) or st.session_state.get('ai_thinking', False)
    )
    
    st.session_state.target_column = target_col
    st.session_state.raw_data = raw_data  # Store for later use in context building
    st.session_state.target_col = target_col  # Store for context builder
    
    
    col1, col2, col3, col4 = st.columns(4)
    target_data = raw_data[target_col]
    with col1:
        st.metric("Mean", f"{target_data.mean():.2f}")
    with col2:
        st.metric("Std Dev", f"{target_data.std():.2f}")
    with col3:
        st.metric("Min", f"{target_data.min():.2f}")
    with col4:
        st.metric("Max", f"{target_data.max():.2f}")
    
    # Data Analytics Section
    st.markdown('<div class="section-header">📊 Step 2.1: Data Analytics</div>', unsafe_allow_html=True)
    
    analytics_expander = st.expander("🔍 Explore Data Analytics & Statistics", expanded=False)
    with analytics_expander:
        try:
            from interface import DataAnalyticsUI
            from services import ContextBuilder
            
            DataAnalyticsUI.render_data_analytics(raw_data, target_col)
            
            # Build and store data analytics context for later use
            if 'data_analytics_context' not in st.session_state:
                st.session_state.data_analytics_context = ContextBuilder.build_data_context(raw_data, target_col)
        except Exception as e:
            st.warning(f"Could not load data analytics: {e}")
            logger.warning(f"Data analytics error: {e}")
    
    # AI Insights Section
    st.markdown('<div class="section-header">🤖 Step 2.2: AI Data Insights</div>', unsafe_allow_html=True)
    
    ai_insights_expander = st.expander("💡 Get AI-Powered Data Insights Before Training", expanded=False)
    with ai_insights_expander:
        try:
            from services import ContextBuilder, ChatInterface
            
            # Build comprehensive data context
            data_context = ContextBuilder.build_data_context(raw_data, target_col)
            
            # AI Provider Selection
            st.markdown("**Select AI Provider:**")
            col1, col2 = st.columns(2)
            with col1:
                ai_provider_option = st.radio(
                    "Choose AI provider:",
                    ["🤖 Gemini", "🔷 ChatGPT"],
                    index=0 if st.session_state.ai_provider == 'gemini' else 1,
                    horizontal=True,
                    disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
                )
                st.session_state.ai_provider = 'gemini' if 'Gemini' in ai_provider_option else 'chatgpt'
            
            with col2:
                api_status = "✅ Available" if st.session_state.ai_provider == 'gemini' and os.getenv("GOOGLE_API_KEY") else "✅ Available" if st.session_state.ai_provider == 'chatgpt' and os.getenv("OPENAI_API_KEY") else "❌ Not Configured"
                st.info(f"**Status:** {api_status}")
            
            # Create or update chat interface with selected provider
            if 'chat_interface' not in st.session_state or st.session_state.get('chat_interface_provider') != st.session_state.ai_provider:
                st.session_state.chat_interface = ChatInterface(ai_provider=st.session_state.ai_provider)
                st.session_state.chat_interface_provider = st.session_state.ai_provider
            
            chat = st.session_state.chat_interface
            
            if chat.ai_available:
                st.info("💡 Ask AI about your data patterns and quality before training:")
                col1, col2 = st.columns([3, 1])
                with col1:
                    data_question = st.text_input(
                        "Ask a question about your data:",
                        placeholder="e.g., What patterns might affect model training? Are there data quality issues?",
                        key="ai_data_insights_input"
                    )
                with col2:
                    if st.button("🔍 Get Insight", key="ai_data_insights_btn"):
                        if data_question:
                            try:
                                with st.spinner("🤔 Generating AI insight... This may take a moment."):
                                    # Generate AI insight
                                    insight = chat.explainer.answer_user_query(data_question, data_context)
                                if insight:
                                    st.success("✅ AI Insight Generated")
                                    st.markdown(insight)
                                    # Save to global history via chat interface
                                    chat.add_message('user', data_question)
                                    chat.add_message('assistant', insight)
                                else:
                                    st.warning("Could not generate insight. Try asking a specific question.")
                            except Exception as e:
                                logger.error(f"Error generating AI insight: {e}")
                                error_msg = str(e)
                                if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
                                    st.error("❌ OpenAI API Quota Exceeded\n\nPlease check your OpenAI account billing and quota limits at https://platform.openai.com/account/billing/overview")
                                elif "429" in error_msg:
                                    st.error("❌ API Rate Limited\n\nToo many requests. Please wait a moment and try again.")
                                else:
                                    st.warning(f"⚠️ Error: {error_msg}")
                        else:
                            st.warning("Please enter a question first!")
            else:
                if st.session_state.ai_provider == 'gemini':
                    st.info("⚠️ AI insights not available. Set GOOGLE_API_KEY environment variable to enable Gemini.")
                else:
                    st.info("⚠️ AI insights not available. Set OPENAI_API_KEY environment variable to enable ChatGPT.")
        except Exception as e:
            st.info(f"AI insights feature not available: {e}")
            logger.debug(f"AI data insights error: {e}")
    
    # Step 3: Training Configuration
    st.markdown('<div class="section-header">⚙️ Step 3: Training Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = 1 - st.slider(
            "Train/Test Split",
            min_value=0.6,
            max_value=0.95,
            value=0.8,
            step=0.05,
            help="Proportion of data used for training",
            disabled=st.session_state.get('training_in_progress', False) or st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
        )
    
    with col2:
        training_type = st.radio(
            "**Training Paradigm**",
            [
                "1️⃣ Standard (No Tuning)",
                "2️⃣ Tuned Only (with CV)",
                "3️⃣ Both (Compare Standard vs Tuned)"
            ],
            help="""
            • **Standard**: Fast training with default parameters (~150-200s)
            • **Tuned Only**: Hyperparameter optimization with CV (~400-600s)
            • **Both**: Train both paradigms in parallel for comparison (~600-900s)
            """,
            disabled=st.session_state.get('training_in_progress', False) or st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
        )
        
        # Store training paradigm in session state for reliable retrieval
        if "Both" in training_type:
            st.session_state['training_paradigm'] = 'Both'
        elif "Tuned" in training_type:
            st.session_state['training_paradigm'] = 'Tuned'
        else:
            st.session_state['training_paradigm'] = 'Standard'
    
    # Training type specific configuration
    if "Both" in training_type:
        st.markdown("**Hyperparameter Tuning Configuration** (for Tuned training)")
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Higher folds = more robust tuning but slower training",
            disabled=st.session_state.get('training_in_progress', False) or st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
        )
        tune_configs = [
            {'tune': False, 'cv_folds': 5, 'name': 'Standard'},
            {'tune': True, 'cv_folds': cv_folds, 'name': 'Tuned'}
        ]
    elif "Tuned" in training_type:
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=5,
            help="Higher folds = more robust tuning but slower training",
            disabled=st.session_state.get('training_in_progress', False) or st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
        )
        tune_configs = [
            {'tune': True, 'cv_folds': cv_folds, 'name': 'Tuned'}
        ]
    else:
        tune_configs = [
            {'tune': False, 'cv_folds': 5, 'name': 'Standard'}
        ]
    
    # Display estimated time
    estimated_times = []
    for config in tune_configs:
        base_time = 150 if not config['tune'] else 400
        estimated_times.append(f"{config['name']}: ~{base_time}s")
    
    st.markdown(f"**Estimated Training Time:** {', '.join(estimated_times)}")
    
    # Step 4: Train
    st.markdown('<div class="section-header">🚀 Step 4: Start Training</div>', unsafe_allow_html=True)
    
    if st.button("🎬 Start Training", type="primary", width='stretch', disabled=st.session_state.get('training_in_progress', False)):
        # Set training in progress flag IMMEDIATELY before any processing
        st.session_state['training_in_progress'] = True
        st.rerun()  # Rerun to update UI with disabled controls
    
    # Check if we should run training
    if st.session_state.get('training_in_progress', False):
        st.info("🚀 Training is in progress... All controls are disabled. Do not refresh the page.")
        try:
            perf_tracker.start_timer("full_pipeline")
            
            data_loader = st.session_state.data_loader
            raw_data = st.session_state.raw_data
            target_column = st.session_state.target_column
            
            # Set target column FIRST (before splitting)
            data_loader.set_target_column(target_column)
            
            # Then split data
            X_train, X_test, y_train, y_test = data_loader.split_train_test(
                test_size=test_size
            )
            
            # Preprocessing
            st.info("🔄 Preprocessing data with 3 techniques...")
            preprocessor = SpectralPreprocessor()
            
            # Preprocess with all 3 techniques
            X_train_proc = {}
            X_test_proc = {}
            for tech in ['reflectance', 'absorbance', 'continuum_removal']:
                X_train_proc[tech] = preprocessor.fit_transform(
                    X_train.values, technique=tech, scaler='standard'
                )
                X_test_proc[tech] = preprocessor.transform(X_test.values)
            
            st.session_state.preprocessor = preprocessor
            
            # Training progress
            progress_placeholder = st.empty()
            results_placeholder = st.empty()
            
            all_results = {}
            
            for config in tune_configs:
                config_name = config['name']
                progress_placeholder.info(f"⏳ Training {config_name} models...")
                
                trainer = ModelTrainer(
                    tune_hyperparameters=config['tune'],
                    cv_folds=config['cv_folds']
                )
                
                # Train all combinations by iterating through techniques
                # and training all models with each technique
                all_technique_results = []
                
                for tech in ['reflectance', 'absorbance', 'continuum_removal']:
                    X_train_tech = X_train_proc[tech]
                    X_test_tech = X_test_proc[tech]
                    y_train_vals = y_train.values
                    y_test_vals = y_test.values
                    
                    # Train all 5 models with this technique
                    for model_name in trainer.models.keys():
                        try:
                            model = trainer.models[model_name]
                            model.train(X_train_tech, y_train_vals)
                            
                            # Get predictions
                            y_train_pred = model.predict(X_train_tech)
                            y_test_pred = model.predict(X_test_tech)
                            
                            # Calculate metrics
                            train_r2 = r2_score(y_train_vals, y_train_pred)
                            test_r2 = r2_score(y_test_vals, y_test_pred)
                            train_rmse = np.sqrt(mean_squared_error(y_train_vals, y_train_pred))
                            test_rmse = np.sqrt(mean_squared_error(y_test_vals, y_test_pred))
                            train_mae = mean_absolute_error(y_train_vals, y_train_pred)
                            test_mae = mean_absolute_error(y_test_vals, y_test_pred)
                            test_rpd = np.std(y_test_vals) / test_rmse if test_rmse > 0 else 0
                            
                            # Extract hyperparameters from the trained model using ParameterInspector
                            hyperparams = {}
                            try:
                                from model_analyzer import ParameterInspector
                                hyperparams = ParameterInspector.get_hyperparameters(model)
                            except Exception as hp_error:
                                logger.debug(f"Could not extract hyperparameters for {model_name}: {hp_error}")
                            
                            all_technique_results.append({
                                'Model': model_name,
                                'Technique': tech,
                                'Train_R²': train_r2,
                                'Test_R²': test_r2,
                                'Train_RMSE': train_rmse,
                                'Test_RMSE': test_rmse,
                                'Train_MAE': train_mae,
                                'Test_MAE': test_mae,
                                'RPD': test_rpd,
                                'Model_Object': model,
                                'Hyperparameters': hyperparams,
                                'Predictions': y_test_pred
                            })
                        except Exception as e:
                            st.warning(f"⚠️ Error training {model_name} with {tech}: {str(e)}")
                            logger.error(f"Error training {model_name} with {tech}: {str(e)}", component="app")
                
                # Create results DataFrame
                results = pd.DataFrame(all_technique_results)
                trainer.results = results  # Store in trainer
                
                # Get the best model (sorted by Test_R²)
                best_idx = results['Test_R²'].idxmax() if len(results) > 0 else 0
                best_result_row = results.iloc[best_idx] if len(results) > 0 else None
                best_model_obj = best_result_row['Model_Object'] if best_result_row is not None else None
                
                all_results[config_name] = {
                    'trainer': trainer,
                    'results': results,
                    'best_model': (best_result_row.to_dict(), best_model_obj) if best_result_row is not None else None
                }
                
                if config_name == 'Standard':
                    st.session_state.trainer_standard = trainer
                elif config_name == 'Tuned':
                    st.session_state.trainer_tuned = trainer
            
            st.session_state.all_training_results = all_results
            st.session_state.training_complete = True
            
            progress_placeholder.empty()
            st.success("✅ Training Complete!")
            
            # Save best models from each paradigm with comprehensive metadata
            try:
                persister = ModelPersistence(model_dir="./saved_models")
                
                # Determine which models to save
                models_to_save = {}
                
                # If both Standard and Tuned exist, only save the better one
                if len(all_results) == 2 and 'Standard' in all_results and 'Tuned' in all_results:
                    standard_data = all_results['Standard']['best_model']
                    tuned_data = all_results['Tuned']['best_model']
                    
                    if standard_data and tuned_data:
                        standard_r2 = float(standard_data[0].get('Test_R²', 0))
                        tuned_r2 = float(tuned_data[0].get('Test_R²', 0))
                        
                        # Save only the better one
                        if standard_r2 >= tuned_r2:
                            models_to_save['Standard'] = standard_data
                            logger.info(f"Using Standard model (R²={standard_r2:.4f} >= Tuned R²={tuned_r2:.4f})")
                        else:
                            models_to_save['Tuned'] = tuned_data
                            logger.info(f"Using Tuned model (R²={tuned_r2:.4f} > Standard R²={standard_r2:.4f})")
                else:
                    # Save all available models (only Standard or only Tuned)
                    for config_name, data in all_results.items():
                        if data['best_model']:
                            models_to_save[config_name] = data['best_model']
                
                # Save selected models
                for config_name, best_data in models_to_save.items():
                    best_result, best_model = best_data
                    
                    # Prepare comprehensive metadata
                    metadata = {
                        'algorithm': str(best_result.get('Model', 'Unknown')),
                        'technique': str(best_result.get('Technique', 'Unknown')),
                        'training_type': config_name,
                        'train_r2': float(best_result.get('Train_R²', 0)),
                        'test_r2': float(best_result.get('Test_R²', 0)),
                        'train_rmse': float(best_result.get('Train_RMSE', 0)),
                        'test_rmse': float(best_result.get('Test_RMSE', 0)),
                        'train_mae': float(best_result.get('Train_MAE', 0)),
                        'test_mae': float(best_result.get('Test_MAE', 0)),
                        'rpd': float(best_result.get('RPD', 0)),
                        'n_features': X_train.shape[1] if 'X_train' in locals() else 0,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'training_pipeline'
                    }
                    
                    # Try to extract and add hyperparameters from the model
                    try:
                        if hasattr(best_model, 'get_params'):
                            hyperparams = best_model.get_params()
                            # Filter and flatten hyperparams for storage
                            filtered_hyperparams = {
                                k: str(v) for k, v in hyperparams.items()
                                if not k.startswith('_') and len(str(v)) < 500
                            }
                            metadata['hyperparameters'] = filtered_hyperparams
                    except Exception as hp_error:
                        logger.debug(f"Could not extract hyperparameters: {hp_error}")
                    
                    # Save model with paradigm in filename
                    persister.save_model(
                        best_model,
                        f"{config_name}_{best_result.get('Model', 'Unknown')}",
                        'regression',
                        best_result.get('Technique', 'Unknown'),
                        metadata=metadata
                    )
            except Exception as save_error:
                logger.warning(f"Could not save best models: {save_error}")
            
            st.rerun()  # Rerun to display results at the top
        
        except Exception as e:
            st.markdown(f'<div class="error-box"><b>❌ Training Error:</b><br>{str(e)}</div>', unsafe_allow_html=True)
            logger.error(f"Training error: {str(e)}", component="app")
        finally:
            # Clear training in progress flag
            st.session_state['training_in_progress'] = False


# ============================================================================
# PREDICTION MODE FUNCTIONS
# ============================================================================

def prediction_mode():
    """Prediction mode interface."""
    st.markdown(
        '<div class="main-header">🔮 Spectral Soil Modeler - PREDICTION MODE</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("""
    **Make predictions using trained models**  
    Load a trained model and get predictions on new spectral data.
    """)
    
    # Check if models exist
    model_dir = Path("./saved_models")
    if not model_dir.exists() or not list(model_dir.glob("*.joblib")):
        st.markdown(
            '<div class="error-box"><b>❌ No Saved Models Found</b><br>'
            'Please train models first in Training Mode. No model files (.joblib) found in ./saved_models/</div>',
            unsafe_allow_html=True
        )
        return
    
    # Step 1: Load Model
    st.markdown('<div class="section-header">🔧 Step 1: Load Trained Model</div>', unsafe_allow_html=True)
    
    # Get available models
    model_files = list(model_dir.glob("*.joblib"))
    if not model_files:
        st.markdown(
            '<div class="error-box"><b>❌ No Models Available</b><br>No .joblib model files found.</div>',
            unsafe_allow_html=True
        )
        return
    
    model_names = [f.stem for f in model_files]
    selected_model = st.selectbox(
        "Select a trained model",
        model_names,
        help="Choose from available trained models",
        disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
    )
    
    try:
        persister = ModelPersistence(model_dir="./saved_models")
        # Parse model name to extract paradigm, algorithm and technique
        parts = selected_model.split('_')
        
        # Try to identify paradigm (Standard or Tuned)
        paradigm = "Unknown"
        algorithm = "Unknown"
        technique = "Unknown"
        
        if len(parts) >= 1:
            # First part might be paradigm (Standard or Tuned)
            if parts[0] in ['Standard', 'Tuned']:
                paradigm = parts[0]
                if len(parts) >= 2:
                    algorithm = parts[1]
                if len(parts) >= 3:
                    technique = '_'.join(parts[2:])
            else:
                # Fallback: first part is algorithm, rest is technique
                algorithm = parts[0]
                if len(parts) >= 2:
                    technique = '_'.join(parts[1:])
        
        # Find the model file - get most recent one matching the pattern
        model_files = list(Path("./saved_models").glob(f"{selected_model}*.joblib"))
        if not model_files:
            st.markdown(
                f'<div class="error-box"><b>❌ Model File Not Found</b><br>'
                f'No model file found for: {selected_model}</div>',
                unsafe_allow_html=True
            )
            return
        
        # Use the most recent file
        filepath = max(model_files, key=lambda p: p.stat().st_mtime)
        
        model = persister.load_model(str(filepath))
        
        # Load metadata from the corresponding JSON file
        # The metadata file should be in model_metadata directory with _metadata.json suffix
        metadata_filename = filepath.stem.replace('.joblib', '') + '_metadata.json'
        
        metadata = None
        try:
            # Try to load metadata
            metadata = persister.load_metadata(metadata_filename)
            logger.info(f"Metadata loaded successfully for {selected_model}")
        except Exception as metadata_error:
            logger.warning(f"Could not load metadata for {selected_model}: {metadata_error}")
            metadata = None
        
        # If no metadata found, create a comprehensive one from the model itself
        if not metadata:
            metadata = {
                'algorithm': algorithm if algorithm != "Unknown" else 'Unknown',
                'technique': technique if technique != "Unknown" else 'Unknown',
                'training_type': paradigm,
                'test_r2': 0.0,
                'train_r2': 0.0,
                'test_rmse': 0.0,
                'train_rmse': 0.0,
                'test_mae': 0.0,
                'train_mae': 0.0,
                'rpd': 0.0,
                'source': 'loaded_from_file_without_metadata',
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Created default metadata for {selected_model}")
        
        st.session_state.loaded_model = model
        st.session_state.model_metadata = metadata
        
        st.success(f"✅ Model loaded: {algorithm}_{technique}")
        
        # Display model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algorithm", metadata.get('algorithm', 'Unknown'))
        with col2:
            st.metric("Technique", metadata.get('technique', 'Unknown'))
        with col3:
            r2_val = metadata.get('test_r2', 0)
            if isinstance(r2_val, (int, float)) and r2_val > 0:
                st.metric("Test R²", f"{r2_val:.4f}")
            else:
                st.metric("Test R²", "Not available")
        
        # Enhanced model inspection section - Display comprehensive model information
        st.markdown('<div class="section-header">� Detailed Model Information</div>', unsafe_allow_html=True)
        
        if ENHANCEMENTS_AVAILABLE:
            try:
                from interface import PredictionModeInspection
                PredictionModeInspection.display_full_model_info(model, metadata)
            except Exception as e:
                st.warning(f"Could not display detailed model inspection: {e}")
                logger.warning(f"Error loading detailed inspection: {e}")
                # Fallback to basic inspection
                try:
                    render_prediction_model_inspection(model, metadata)
                except:
                    st.info("Model inspection not available")
        else:
            st.info("Advanced model inspection features are not available.")
    
    except FileNotFoundError:
        st.markdown(
            f'<div class="error-box"><b>❌ Model Not Found</b><br>'
            f'Could not load model: {selected_model}</div>',
            unsafe_allow_html=True
        )
        return
    except Exception as e:
        st.markdown(
            f'<div class="error-box"><b>❌ Error Loading Model</b><br>{str(e)}</div>',
            unsafe_allow_html=True
        )
        logger.error(f"Error loading model: {str(e)}", component="app")
        return
    
    # Step 2: Upload Prediction Data
    st.markdown('<div class="section-header">📊 Step 2: Upload Data for Prediction</div>', unsafe_allow_html=True)
    
    pred_file = st.file_uploader(
        "Upload spectral data for prediction (XLS/CSV)",
        type=['xlsx', 'csv', 'xls'],
        help="File should contain the same spectral features as training data",
        key="pred_file_upload",
        disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
    )
    
    if not pred_file:
        st.info("👆 Upload spectral data file to make predictions")
        return
    
    try:
        temp_pred_path = f"./temp_pred_{pred_file.name}"
        with open(temp_pred_path, "wb") as f:
            f.write(pred_file.getbuffer())
        
        pred_loader = DataLoader()
        pred_data = pred_loader.load_data(temp_pred_path)
        
        st.success(f"✅ Data loaded: {pred_data.shape[0]} samples, {pred_data.shape[1]} features")
        
        with st.expander("View Prediction Data Sample"):
            st.dataframe(pred_data.head(10), width='stretch')
    
    except Exception as e:
        st.markdown(
            f'<div class="error-box"><b>❌ Error Loading Prediction Data</b><br>{str(e)}</div>',
            unsafe_allow_html=True
        )
        logger.error(f"Error loading prediction data: {str(e)}", component="app")
        return
    
    # Step 3: Automatic Preprocessing Detection
    st.markdown('<div class="section-header">⚙️ Step 3: Preprocessing Configuration</div>', unsafe_allow_html=True)
    
    # Extract preprocessing technique from model metadata or filename
    technique = metadata.get('technique', 'reflectance').lower() if metadata else 'reflectance'
    
    # Normalize technique name
    if technique in ['reflectance']:
        technique = 'reflectance'
    elif technique in ['absorbance']:
        technique = 'absorbance'
    elif technique in ['continuum_removal', 'continuum removal', 'continuum_removed']:
        technique = 'continuum_removal'
    else:
        technique = 'reflectance'  # Default
    
    st.info(f"""
    **✅ Preprocessing Configuration Detected:**
    - **Technique:** {technique.replace('_', ' ').title()}
    - **Source:** Extracted from model metadata
    
    This preprocessing will be automatically applied to your prediction data to match the training conditions.
    """)
    
    # Initialize preprocessor based on detected technique
    preprocessor = SpectralPreprocessor()
    st.session_state.loaded_preprocessor = preprocessor
    
    # Step 4: Make Predictions
    st.markdown('<div class="section-header">🚀 Step 4: Generate Predictions</div>', unsafe_allow_html=True)
    
    if st.button("🔮 Make Predictions", type="primary", width='stretch', disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)):
        try:
            loaded_model = st.session_state.loaded_model
            loaded_preprocessor = st.session_state.loaded_preprocessor
            
            # Preprocess prediction data using the detected technique
            if loaded_preprocessor:
                try:
                    # Fit the preprocessor with the technique used during training
                    X_pred = loaded_preprocessor.fit_transform(pred_data.values, technique=technique)
                except Exception as e:
                    logger.warning(f"Could not apply technique-based preprocessing: {e}, using basic scaling")
                    # Fallback to basic scaling
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_pred = scaler.fit_transform(pred_data.values)
            else:
                # Use default preprocessing
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_pred = scaler.fit_transform(pred_data.values)
            
            # Make predictions
            predictions = loaded_model.predict(X_pred)
            
            # Display results
            st.markdown('<div class="section-header">📈 Prediction Results</div>', unsafe_allow_html=True)
            
            results_df = pd.DataFrame({
                'Sample_ID': range(1, len(predictions) + 1),
                'Predicted_Value': predictions,
                'Uncertainty': np.abs(np.gradient(predictions))  # Simple uncertainty measure
            })
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Prediction", f"{predictions.mean():.2f}")
            with col2:
                st.metric("Std Dev", f"{predictions.std():.2f}")
            with col3:
                st.metric("Total Samples", len(predictions))
            
            st.dataframe(results_df, width='stretch')
            
            # Visualization
            fig = px.line(
                results_df,
                x='Sample_ID',
                y='Predicted_Value',
                title='Predicted Values Across Samples',
                labels={'Predicted_Value': 'Predicted Soil Property', 'Sample_ID': 'Sample ID'}
            )
            st.plotly_chart(fig, width='stretch')
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Predictions generated successfully!")
        
        except Exception as e:
            st.markdown(
                f'<div class="error-box"><b>❌ Prediction Error</b><br>{str(e)}</div>',
                unsafe_allow_html=True
            )
            logger.error(f"Prediction error: {str(e)}", component="app")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main app function with mode selection."""
    init_session_state()
    
    # Sidebar mode selection
    with st.sidebar:
        st.markdown("## 🎯 SELECT MODE")
        st.markdown("---")
        
        # Check if any background operation is happening
        is_busy = st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
        
        # Determine default index based on session state
        mode_options = ["🚀 Training Mode", "🔮 Prediction Mode"]
        default_idx = 0 if st.session_state.app_mode == 'training' else 1
        
        mode = st.radio(
            "Choose Application Mode:",
            mode_options,
            index=default_idx,
            help="Training: Build and compare models\nPrediction: Use trained models for inference",
            key="app_mode_selector",
            disabled=st.session_state.get('training_in_progress', False) or is_busy
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Management")
        
        # Disable model management section during operations
        if is_busy:
            st.info("⏳ Please wait - Operation in progress...")
        
        model_dir = Path("./saved_models")
        if model_dir.exists():
            model_count = len(list(model_dir.glob("*.joblib")))
            st.metric("Saved Models", model_count)
        else:
            st.metric("Saved Models", 0)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Spectral Soil Modeler**
        
        An automated ML pipeline for predicting soil properties from spectral data.
        
        **Features:**
        - 3 preprocessing techniques
        - 5 ML algorithms
        - 3 training paradigms
        - Automated hyperparameter tuning
        - Model comparison & visualization
        """)
    
    # Route to appropriate mode
    if "Training" in mode:
        st.session_state.app_mode = 'training'
        training_mode()
    else:
        st.session_state.app_mode = 'prediction'
        prediction_mode()


if __name__ == "__main__":
    main()
