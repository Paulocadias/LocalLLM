"""
Weekly Complete Improvement System

Runs automatically every week (Monday 2 AM):
1. Tests current models vs commercial AI (10 tests)
2. Generates reflexion training data
3. Creates improved models
4. Accumulates training data for monthly fine-tuning
5. Generates weekly report
6. Tracks long-term evolution

Set and Forget - runs automatically!
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add model-improvement-service to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model-improvement-service'))

from automatic_tester import AutomaticTester

# Configuration
CLAUDE_API_KEY = os.getenv("COMMERCIAL_AI_API_KEY")
if not CLAUDE_API_KEY:
    print("ERROR: COMMERCIAL_AI_API_KEY environment variable must be set")
    print("Get your API key from: https://console.anthropic.com/")
    sys.exit(1)
MODELS_TO_TEST = [
    "qwen-coder-enhanced",
    "qwen2.5-coder:latest",
    "deepseek-coder-v2:latest"
]
TESTS_PER_MODEL = 10
WEEKLY_RESULTS_DIR = Path("weekly_results")
TRAINING_DATA_DIR = Path("datasets/lora_profiles")
HISTORY_FILE = Path("weekly_improvement_history.json")

# Email configuration (optional)
SEND_EMAIL = os.getenv("SEND_EMAIL_REPORTS", "false").lower() == "true"
EMAIL_TO = os.getenv("REPORT_EMAIL", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# Create directories
WEEKLY_RESULTS_DIR.mkdir(exist_ok=True)
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_history():
    """Load historical improvement data"""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return {"weeks": []}


def save_history(history):
    """Save historical improvement data"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def test_all_models(tester: AutomaticTester, week_num: int):
    """Test all configured models"""

    print()
    print("="*80)
    print(f"WEEK {week_num} - TESTING ALL MODELS")
    print("="*80)
    print()

    week_results = {
        "week": week_num,
        "date": datetime.now().isoformat(),
        "models": []
    }

    for model_name in MODELS_TO_TEST:
        print(f"\nTesting model: {model_name}")
        print("-" * 60)

        try:
            # Run comparison test
            comparison_results = tester.run_comparison_test(model_name, TESTS_PER_MODEL)
            stats = comparison_results["statistics"]

            # Calculate average scores
            total_scores = {}
            for test in comparison_results["tests"]:
                for dimension, scores in test["quality_metrics"].items():
                    if isinstance(scores, dict) and "localllm" in scores:
                        total_scores.setdefault(dimension, []).append(scores["localllm"])

            avg_scores = {
                dim: sum(scores) / len(scores) if scores else 0
                for dim, scores in total_scores.items()
            }

            # Generate training data
            reflexion_examples = tester.generate_reflexion_training(comparison_results)
            distillation_examples = tester.generate_distillation_training(comparison_results)

            # Save training data
            reflexion_file, distillation_file = tester.save_training_data(
                f"{model_name}_week{week_num}",
                reflexion_examples,
                distillation_examples
            )

            # Create improved model
            improved_model = f"{model_name.replace(':', '-')}-week{week_num}"
            success = tester.create_improved_model(model_name, reflexion_file, improved_model)

            # Record results
            model_result = {
                "model_name": model_name,
                "improved_model": improved_model if success else None,
                "win_rate": stats["win_rate"],
                "wins": stats["wins"],
                "ties": stats["ties"],
                "losses": stats["losses"],
                "average_scores": avg_scores,
                "reflexion_examples": len(reflexion_examples),
                "distillation_examples": len(distillation_examples),
                "comparison_file": str(comparison_results.get("comparison_file", ""))
            }

            week_results["models"].append(model_result)

            print(f"\n{model_name} Results:")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Average Quality: {sum(avg_scores.values()) / len(avg_scores):.1f}/10")
            print(f"  Training Examples: {len(reflexion_examples)} reflexion")

        except Exception as e:
            print(f"\nERROR testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save week results
    week_file = WEEKLY_RESULTS_DIR / f"week_{week_num}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(week_file, 'w') as f:
        json.dump(week_results, f, indent=2)

    return week_results


def generate_weekly_report(week_results, history):
    """Generate weekly improvement report"""

    report = []
    report.append("="*80)
    report.append(f"WEEKLY IMPROVEMENT REPORT - Week {week_results['week']}")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("="*80)
    report.append("")

    # Current week results
    report.append("THIS WEEK'S RESULTS")
    report.append("-" * 60)

    for model in week_results["models"]:
        report.append(f"\nModel: {model['model_name']}")
        report.append(f"  Win Rate: {model['win_rate']:.1f}%")
        report.append(f"  Wins: {model['wins']}, Ties: {model['ties']}, Losses: {model['losses']}")

        avg_quality = sum(model['average_scores'].values()) / len(model['average_scores'])
        report.append(f"  Average Quality: {avg_quality:.1f}/10")
        report.append(f"  Training Examples Generated: {model['reflexion_examples']}")

        if model['improved_model']:
            report.append(f"  Improved Model Created: {model['improved_model']}")

    # Historical comparison
    if len(history["weeks"]) > 1:
        report.append("")
        report.append("="*80)
        report.append("EVOLUTION OVER TIME")
        report.append("="*80)
        report.append("")

        # Track each model over time
        for model_name in MODELS_TO_TEST:
            report.append(f"\n{model_name} Evolution:")
            report.append("-" * 60)

            for week_data in history["weeks"][-4:]:  # Last 4 weeks
                for model in week_data.get("models", []):
                    if model["model_name"] == model_name:
                        week_num = week_data["week"]
                        win_rate = model["win_rate"]
                        bar = "#" * int(win_rate / 2)
                        report.append(f"  Week {week_num:2d}: {win_rate:5.1f}% [{bar}]")

        # Calculate improvement trend
        report.append("")
        report.append("IMPROVEMENT TREND")
        report.append("-" * 60)

        for model_name in MODELS_TO_TEST:
            model_history = []
            for week_data in history["weeks"]:
                for model in week_data.get("models", []):
                    if model["model_name"] == model_name:
                        model_history.append(model)

            if len(model_history) >= 2:
                first = model_history[0]
                latest = model_history[-1]

                win_rate_change = latest["win_rate"] - first["win_rate"]

                first_quality = sum(first["average_scores"].values()) / len(first["average_scores"])
                latest_quality = sum(latest["average_scores"].values()) / len(latest["average_scores"])
                quality_change = latest_quality - first_quality

                report.append(f"\n{model_name}:")
                report.append(f"  Win Rate: {first['win_rate']:.1f}% -> {latest['win_rate']:.1f}% ({win_rate_change:+.1f}%)")
                report.append(f"  Quality: {first_quality:.1f} -> {latest_quality:.1f} ({quality_change:+.1f} pts)")

                if win_rate_change > 5:
                    report.append(f"  Status: [SUCCESS] Significant improvement!")
                elif win_rate_change > 0:
                    report.append(f"  Status: [SUCCESS] Improving")
                else:
                    report.append(f"  Status: [STABLE] Continue training")

    # Training data accumulation
    report.append("")
    report.append("="*80)
    report.append("TRAINING DATA ACCUMULATED")
    report.append("="*80)

    total_reflexion = sum(
        model["reflexion_examples"]
        for week_data in history["weeks"]
        for model in week_data.get("models", [])
    )

    total_distillation = sum(
        model["distillation_examples"]
        for week_data in history["weeks"]
        for model in week_data.get("models", [])
    )

    report.append(f"Total Reflexion Examples: {total_reflexion}")
    report.append(f"Total Distillation Examples: {total_distillation}")
    report.append(f"Total Training Examples: {total_reflexion + total_distillation}")

    if total_reflexion + total_distillation >= 100:
        report.append("")
        report.append("[READY] Enough training data for fine-tuning!")
        report.append("Consider running monthly fine-tuning to apply improvements.")

    report.append("")
    report.append("="*80)
    report.append("Next run: Next Monday at 2 AM")
    report.append("="*80)

    return "\n".join(report)


def send_email_report(subject, body):
    """Send email report (optional)"""

    if not SEND_EMAIL or not EMAIL_TO:
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_FROM, EMAIL_TO, text)
        server.quit()

        print("[EMAIL] Report sent successfully")

    except Exception as e:
        print(f"[EMAIL] Failed to send report: {e}")


def main():
    """Run weekly improvement"""

    print("="*80)
    print("WEEKLY COMPLETE IMPROVEMENT SYSTEM")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Load history
    history = load_history()
    week_num = len(history["weeks"]) + 1

    print(f"Running Week {week_num} improvement...")
    print(f"Models to test: {', '.join(MODELS_TO_TEST)}")
    print(f"Tests per model: {TESTS_PER_MODEL}")
    print()

    # Initialize tester
    tester = AutomaticTester(claude_api_key=CLAUDE_API_KEY)

    # Test all models
    week_results = test_all_models(tester, week_num)

    # Add to history
    history["weeks"].append(week_results)
    save_history(history)

    # Generate report
    report = generate_weekly_report(week_results, history)

    # Print report
    print()
    print(report)

    # Save report to file
    report_file = WEEKLY_RESULTS_DIR / f"report_week_{week_num}_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")

    # Send email report (if configured)
    if SEND_EMAIL:
        send_email_report(f"LocalLLM Weekly Improvement Report - Week {week_num}", report)

    print()
    print("="*80)
    print("Weekly improvement complete!")
    print("="*80)


if __name__ == "__main__":
    main()
