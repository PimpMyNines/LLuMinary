#!/usr/bin/env python3
"""
Metrics Example for LLM Handler

This example demonstrates how to:
1. Collect and track metrics from LLM interactions
2. Analyze metrics for cost and performance
3. Set up alerts based on metric thresholds
4. View metrics over time

To run this example, make sure you have your API keys set in the environment:
- OPENAI_API_KEY for OpenAI models
- ANTHROPIC_API_KEY for Anthropic models
"""

import json
import os

# Add parent directory to path to import lluminary
import sys
import time
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lluminary import get_llm_from_model
from monitoring.alerting import check_alerts, set_alert_threshold
from monitoring.analytics import analyze_metrics, collect_metrics, get_metrics_over_time

# Set API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def collect_model_metrics(model_name, api_key, num_requests=5):
    """Collect metrics from multiple requests to a model."""
    print(f"\n--- Collecting metrics from {model_name} ---")

    # Initialize the model
    llm = get_llm_from_model(model_name, api_key=api_key)

    # Storage for metrics
    all_metrics = []
    total_cost = 0
    total_tokens = 0
    response_times = []

    # Make multiple requests to gather metrics
    for i in range(num_requests):
        start_time = time.time()

        # Generate a response
        response, usage, metrics = llm.generate(
            event_id=f"metrics_test_{i}",
            system_prompt="You are a helpful assistant that provides concise answers.",
            messages=[
                {
                    "message_type": "human",
                    "message": f"Explain the concept of {['AI', 'machine learning', 'neural networks', 'natural language processing', 'computer vision'][i % 5]} briefly.",
                }
            ],
            max_tokens=100,
        )

        # Calculate response time
        response_time = time.time() - start_time

        # Add metrics
        metrics_data = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "input_tokens": usage["read_tokens"],
            "output_tokens": usage["write_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost": usage["total_cost"],
            "response_time": response_time,
            "event_id": f"metrics_test_{i}",
        }

        # Store metrics
        all_metrics.append(metrics_data)
        total_cost += usage["total_cost"]
        total_tokens += usage["total_tokens"]
        response_times.append(response_time)

        # Print summary of this request
        print(
            f"Request {i+1}: {metrics_data['total_tokens']} tokens, ${metrics_data['cost']:.6f}, {metrics_data['response_time']:.2f}s"
        )

        # Save metrics to database/storage (simulated)
        collect_metrics(metrics_data)

        # Small delay between requests
        time.sleep(0.5)

    # Calculate and return aggregated metrics
    return {
        "model": model_name,
        "requests": num_requests,
        "avg_tokens": total_tokens / num_requests,
        "total_tokens": total_tokens,
        "avg_cost": total_cost / num_requests,
        "total_cost": total_cost,
        "avg_response_time": sum(response_times) / len(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "detailed_metrics": all_metrics,
    }


def analyze_model_performance(metrics):
    """Analyze performance metrics for a model."""
    print(f"\n--- Performance Analysis for {metrics['model']} ---")
    print(f"Average response time: {metrics['avg_response_time']:.2f}s")
    print(f"Average tokens per request: {metrics['avg_tokens']:.1f}")
    print(f"Average cost per request: ${metrics['avg_cost']:.6f}")
    print(
        f"Total cost for {metrics['requests']} requests: ${metrics['total_cost']:.6f}"
    )

    # Analyze metrics (simulated)
    analysis = analyze_metrics(metrics["detailed_metrics"])

    # Set up alerts based on thresholds
    set_alert_threshold("response_time", 3.0)  # Alert if response time > 3 seconds
    set_alert_threshold("total_cost", 0.10)  # Alert if daily cost > $0.10

    # Check if any metrics trigger alerts
    alerts = check_alerts(metrics["detailed_metrics"])
    if alerts:
        print("\nAlerts triggered:")
        for alert in alerts:
            print(
                f"- {alert['metric']}: {alert['value']} (threshold: {alert['threshold']})"
            )
    else:
        print("\nNo alerts triggered")


def visualize_metrics(openai_metrics, anthropic_metrics=None):
    """Visualize metrics comparison between models."""
    print("\n--- Visualizing Metrics ---")

    # In a notebook, you would visualize with matplotlib
    print("In a notebook, this would create comparison charts showing:")
    print("1. Response time comparison")
    print("2. Token usage comparison")
    print("3. Cost comparison")

    # Example of how to visualize (code commented out as it requires display)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Models to compare
    models = [openai_metrics['model']]
    avg_response_times = [openai_metrics['avg_response_time']]
    avg_tokens = [openai_metrics['avg_tokens']]
    avg_costs = [openai_metrics['avg_cost']]

    if anthropic_metrics:
        models.append(anthropic_metrics['model'])
        avg_response_times.append(anthropic_metrics['avg_response_time'])
        avg_tokens.append(anthropic_metrics['avg_tokens'])
        avg_costs.append(anthropic_metrics['avg_cost'])

    # Plot response times
    ax1.bar(models, avg_response_times)
    ax1.set_ylabel('Average Response Time (s)')
    ax1.set_title('Response Time Comparison')

    # Plot token usage
    ax2.bar(models, avg_tokens)
    ax2.set_ylabel('Average Tokens per Request')
    ax2.set_title('Token Usage Comparison')

    # Plot costs
    ax3.bar(models, avg_costs)
    ax3.set_ylabel('Average Cost per Request ($)')
    ax3.set_title('Cost Comparison')

    plt.tight_layout()
    plt.show()
    """


def track_metrics_over_time():
    """Demonstrate how to track metrics over time."""
    print("\n--- Tracking Metrics Over Time ---")

    # In a real application, you would query your metrics database

    # Simulate metrics over the past week
    now = datetime.now()
    dates = [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    daily_costs = [0.05, 0.08, 0.03, 0.07, 0.06, 0.09, 0.04]
    daily_tokens = [5000, 8000, 3000, 7000, 6000, 9000, 4000]

    # Print sample data
    print("Date\t\tCost\tTokens")
    for date, cost, tokens in zip(dates, daily_costs, daily_tokens):
        print(f"{date}\t${cost:.2f}\t{tokens}")

    # Get metrics over time (simulated)
    time_metrics = get_metrics_over_time(now - timedelta(days=7), now)

    # Example of forecasting usage and costs
    forecast_days = 30
    avg_daily_cost = sum(daily_costs) / len(daily_costs)
    forecast_monthly_cost = avg_daily_cost * forecast_days

    print(f"\nProjected 30-day cost: ${forecast_monthly_cost:.2f}")

    # Example of how you would visualize this data in a notebook
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, daily_costs, marker='o')
    plt.title('Daily API Costs')
    plt.xlabel('Date')
    plt.ylabel('Cost ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """


def export_metrics_to_json(metrics, filename="metrics_export.json"):
    """Export collected metrics to a JSON file for further analysis."""
    print(f"\n--- Exporting Metrics to {filename} ---")

    with open(filename, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics exported successfully to {filename}")


def main():
    """Run the metrics example."""
    print("=== LLM Handler Metrics Example ===\n")

    all_model_metrics = {}

    # Test OpenAI model if API key is available
    if OPENAI_API_KEY:
        openai_metrics = collect_model_metrics(
            "gpt-3.5-turbo", OPENAI_API_KEY, num_requests=3
        )
        analyze_model_performance(openai_metrics)
        all_model_metrics["gpt-3.5-turbo"] = openai_metrics
    else:
        print("Skipping OpenAI model test - API key not set")
        openai_metrics = None

    # Test Anthropic model if API key is available
    if ANTHROPIC_API_KEY:
        anthropic_metrics = collect_model_metrics(
            "claude-haiku-3.5", ANTHROPIC_API_KEY, num_requests=3
        )
        analyze_model_performance(anthropic_metrics)
        all_model_metrics["claude-haiku-3.5"] = anthropic_metrics
    else:
        print("Skipping Anthropic model test - API key not set")
        anthropic_metrics = None

    # Visualize metrics comparison
    if openai_metrics or anthropic_metrics:
        visualize_metrics(
            openai_metrics if openai_metrics else anthropic_metrics,
            anthropic_metrics if openai_metrics else None,
        )

    # Demonstrate tracking metrics over time
    track_metrics_over_time()

    # Export metrics to JSON for further analysis
    if all_model_metrics:
        export_metrics_to_json(all_model_metrics)

    print("\n=== Metrics Example Complete ===")


if __name__ == "__main__":
    main()
