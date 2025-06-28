# Formula 1 Prompting Strategies Experiment

## Overview
This project demonstrates three different prompting strategies (Direct, Few-Shot, Chain-of-Thought) applied to Formula 1 race prediction and compares their effectiveness.

## Task Description
**Objective**: Predict the top 3 finishers in a Monaco Grand Prix based on:
- Qualifying results
- Track characteristics  
- Weather conditions
- Historical data

## Files Structure
```
assignment4/
├── f1_prompts.py              # Three prompting strategies implementation
├── experiment_runner.py       # Experiment execution and analysis
├── prompting_analysis_report.md # Detailed written analysis
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

## Three Prompting Strategies

### 1. Direct Prompting
- **Approach**: Straightforward instruction
- **Pros**: Quick, efficient, low token usage
- **Cons**: Limited reasoning depth
- **Best for**: Simple tasks, quick predictions

### 2. Few-Shot Prompting  
- **Approach**: Provide examples before the actual task
- **Pros**: Good pattern recognition, contextual learning
- **Cons**: Requires quality examples, medium token usage
- **Best for**: Pattern-based tasks, learning new domains

### 3. Chain-of-Thought Prompting
- **Approach**: Step-by-step systematic analysis
- **Pros**: Comprehensive reasoning, transparent logic
- **Cons**: Higher token usage, slower execution
- **Best for**: Complex analysis, high-stakes decisions

## How to Run

1. **Run the complete experiment**:
   ```bash
   python experiment_runner.py
   ```

2. **View individual prompts**:
   ```bash
   python f1_prompts.py
   ```

3. **Read the detailed analysis**:
   ```bash
   # Open prompting_analysis_report.md
   ```

## Expected Output
The experiment will show:
- All three prompts formatted for display
- Simulated LLM responses for each strategy
- Quantitative analysis of prompt complexity
- Response quality metrics
- Rankings and recommendations

## Key Findings

🏆 **Winner: Chain-of-Thought Prompting**

**Why it performed best:**
- Most comprehensive F1-specific analysis
- Incorporated Monaco track characteristics (65% pole win rate)
- Systematic consideration of all variables
- Transparent reasoning process
- Highest accuracy potential

**Metrics:**
- Detail Score: 27.4 (highest)
- F1 Specificity: 8 terms
- Reasoning Indicators: 6 

## Real-World Applications

This experiment demonstrates prompting strategies applicable to:
- **Sports Analytics**: Game/match predictions
- **Financial Forecasting**: Market predictions with multiple variables
- **Medical Diagnosis**: Systematic symptom analysis
- **Business Strategy**: Multi-factor decision making

## Extension Ideas

1. **API Integration**: Connect to actual LLM APIs (OpenAI, Anthropic)
2. **Real Data**: Use live F1 qualifying results and race outcomes
3. **Accuracy Tracking**: Compare predictions to actual race results
4. **Dynamic Examples**: Auto-generate few-shot examples from historical data
5. **Multi-Modal**: Include weather radar, telemetry data
6. **Interactive Dashboard**: Web interface for live predictions

## Educational Value

This project teaches:
- **Prompt Engineering**: Different strategies for different use cases
- **Comparative Analysis**: Systematic evaluation of AI outputs  
- **Domain Application**: Applying AI to specialized knowledge areas
- **Critical Thinking**: Understanding when and why different approaches work

## Contact & Contributions

Feel free to extend this experiment with:
- Additional prompting strategies (role-playing, tree-of-thought)
- Different domains (NBA, stock market, weather)
- Real LLM API integrations
- Statistical validation methods
