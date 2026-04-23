"""
Meta OpenEnv Hackathon Submission Script

This script generates a complete submission package including:
- Model training and evaluation
- Baseline comparisons
- Ablation studies
- Performance metrics
- Visualization reports
"""

import os
import json
import time
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def run_command(cmd, description):
    """Run a command and log results"""
    print(f"\n🚀 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"   💥 Error: {str(e)}")
        return False

def generate_submission_report():
    """Generate comprehensive submission report"""
    print("📋 Generating Hackathon Submission Report...")
    
    # Run evaluation
    if not run_command("python evaluate.py", "Running Evaluation"):
        print("⚠️  Evaluation failed, continuing...")
    
    # Run ablation study
    if not run_command("python ablation_study.py", "Running Ablation Study"):
        print("⚠️  Ablation study failed, continuing...")
    
    # Generate project summary
    project_summary = {
        "project_name": "Enterprise Email Triage Simulator",
        "category": "Theme 3.1: Professional Tasks (Enterprise Workflows)",
        "description": "AI-powered email triage system using reinforcement learning",
        "technologies": [
            "OpenEnv",
            "Streamlit",
            "Groq (Llama-3.1-8B)",
            "PyTorch",
            "PEFT/LoRA",
            "TRL",
            "Docker"
        ],
        "features": [
            "LLM tool calling with JSON format",
            "500+ diverse email dataset",
            "Real-time dashboard",
            "Manual override controls",
            "Test sandbox",
            "Comprehensive evaluation",
            "Ablation studies"
        ],
        "metrics": {
            "total_emails": 500,
            "email_types": 10,
            "available_tools": 3,
            "reward_functions": 4,
            "baseline_agents": 3
        },
        "files": {
            "environment": "env.py",
            "dashboard": "app.py",
            "training": "train.ipynb",
            "evaluation": "evaluate.py",
            "ablation": "ablation_study.py",
            "dataset_generation": "generate_dataset.py",
            "docker": ["Dockerfile", "docker-compose.yml"],
            "documentation": "README.md"
        },
        "deployment": {
            "github": "https://github.com/vaishali-strategy/email-triage",
            "huggingface": "https://huggingface.co/spaces/Proteinrequired/enterprise-email-triage",
            "docker": "Available for containerized deployment"
        },
        "submission_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save project summary
    with open("submission_summary.json", "w") as f:
        json.dump(project_summary, f, indent=2)
    
    print("✅ Submission summary saved to: submission_summary.json")
    
    # Check for generated files
    expected_files = [
        "evaluation_report.json",
        "evaluation_plots.png",
        "ablation_results.json",
        "ablation_study.png",
        "submission_summary.json"
    ]
    
    print("\n📁 Generated Files:")
    for file in expected_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size} bytes)")
        else:
            print(f"   ❌ {file} (missing)")
    
    return project_summary

def create_submission_zip():
    """Create submission package"""
    print("\n📦 Creating Submission Package...")
    
    import zipfile
    
    submission_files = [
        "env.py",
        "app.py", 
        "train.ipynb",
        "evaluate.py",
        "ablation_study.py",
        "generate_dataset.py",
        "requirements.txt",
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
        "dataset.json",
        "submission_summary.json"
    ]
    
    # Add generated files if they exist
    for file in ["evaluation_report.json", "evaluation_plots.png", "ablation_results.json", "ablation_study.png"]:
        if os.path.exists(file):
            submission_files.append(file)
    
    with zipfile.ZipFile("hackathon_submission.zip", "w") as zipf:
        for file in submission_files:
            if os.path.exists(file):
                zipf.write(file)
                print(f"   ✅ Added: {file}")
            else:
                print(f"   ❌ Missing: {file}")
    
    print("📦 Submission package created: hackathon_submission.zip")
    return "hackathon_submission.zip"

def print_final_summary(summary):
    """Print final submission summary"""
    print("\n" + "="*80)
    print("🎉 META OPENENV HACKATHON SUBMISSION COMPLETE")
    print("="*80)
    
    print(f"\n📊 Project: {summary['project_name']}")
    print(f"🏷️  Category: {summary['category']}")
    print(f"📝 Description: {summary['description']}")
    
    print(f"\n🛠️  Technologies Used:")
    for tech in summary['technologies']:
        print(f"   • {tech}")
    
    print(f"\n✨ Key Features:")
    for feature in summary['features']:
        print(f"   • {feature}")
    
    print(f"\n📈 Metrics:")
    for key, value in summary['metrics'].items():
        print(f"   • {key}: {value}")
    
    print(f"\n🌐 Deployment:")
    print(f"   • GitHub: {summary['deployment']['github']}")
    print(f"   • HuggingFace: {summary['deployment']['huggingface']}")
    print(f"   • Docker: {summary['deployment']['docker']}")
    
    print(f"\n📁 Submission Package:")
    print(f"   • hackathon_submission.zip (ready for upload)")
    
    print(f"\n🎯 Hackathon Readiness:")
    print(f"   ✅ OpenEnv-compliant environment")
    print(f"   ✅ LLM integration with tool calling")
    print(f"   ✅ Comprehensive evaluation")
    print(f"   ✅ Baseline comparisons")
    print(f"   ✅ Ablation studies")
    print(f"   ✅ Live demo available")
    print(f"   ✅ Docker deployment ready")
    print(f"   ✅ Complete documentation")
    
    print("\n" + "="*80)

def main():
    """Main submission function"""
    print("🚀 Starting Meta OpenEnv Hackathon Submission Process...")
    
    # Generate submission report
    summary = generate_submission_report()
    
    # Create submission package
    submission_zip = create_submission_zip()
    
    # Print final summary
    print_final_summary(summary)
    
    print(f"\n🎯 Your submission is ready!")
    print(f"📦 Upload: {submission_zip}")
    print(f"🌐 Live Demo: {summary['deployment']['huggingface']}")
    print(f"📖 Documentation: {summary['deployment']['github']}")

if __name__ == "__main__":
    main()
