"""
Formula 1 Race Prediction Prompts
Three different prompting strategies for the same task:
Predicting race outcomes based on qualifying positions and track characteristics
"""

# Sample data for our prompts
f1_data = {
    "qualifying_results": [
        "1. Max Verstappen (Red Bull) - 1:20.374",
        "2. Charles Leclerc (Ferrari) - 1:20.596", 
        "3. Sergio Perez (Red Bull) - 1:20.789",
        "4. Lewis Hamilton (Mercedes) - 1:20.912",
        "5. George Russell (Mercedes) - 1:21.045"
    ],
    "track": "Monaco Grand Prix",
    "track_characteristics": "Tight street circuit, difficult overtaking, high importance of qualifying position",
    "weather": "Sunny, 24°C, no rain expected",
    "historical_note": "In Monaco, the pole sitter wins 65% of the time"
}

def direct_prompt():
    """Direct prompting approach - straightforward instruction"""
    return f"""
Task: Predict the top 3 finishers for the Monaco Grand Prix based on the following data:

Qualifying Results:
{chr(10).join(f1_data['qualifying_results'])}

Track: {f1_data['track']}
Weather: {f1_data['weather']}
Track Info: {f1_data['track_characteristics']}

Predict the top 3 race finishers and briefly explain your reasoning.
"""

def few_shot_prompt():
    """Few-shot prompting approach - providing examples"""
    return f"""
Task: Predict Formula 1 race outcomes based on qualifying results and track characteristics.

Here are some examples:

Example 1:
Qualifying: 1. Hamilton 2. Verstappen 3. Leclerc
Track: Silverstone (high-speed, multiple overtaking opportunities)
Weather: Dry
Prediction: 1. Verstappen 2. Hamilton 3. Leclerc
Reasoning: Verstappen's superior race pace typically allows him to overtake from P2, Hamilton strong at home track.

Example 2:
Qualifying: 1. Leclerc 2. Sainz 3. Verstappen  
Track: Monaco (street circuit, very difficult to overtake)
Weather: Dry
Prediction: 1. Leclerc 2. Sainz 3. Verstappen
Reasoning: Monaco heavily favors qualifying position, Ferrari 1-2 likely to hold.

Example 3:
Qualifying: 1. Russell 2. Verstappen 3. Hamilton
Track: Spa (long straights, DRS zones, rain possible)
Weather: Mixed conditions expected
Prediction: 1. Verstappen 2. Hamilton 3. Russell
Reasoning: Verstappen excels in changing conditions, Hamilton's experience advantage in mixed weather.

Now predict for this race:

Qualifying Results:
{chr(10).join(f1_data['qualifying_results'])}

Track: {f1_data['track']}
Weather: {f1_data['weather']}
Track Info: {f1_data['track_characteristics']}
Historical Note: {f1_data['historical_note']}

Predict the top 3 race finishers and explain your reasoning.
"""

def chain_of_thought_prompt():
    """Chain-of-thought prompting approach - step-by-step reasoning"""
    return f"""
Task: Predict Formula 1 race outcomes using systematic analysis.

Please analyze this step-by-step:

Step 1: Analyze the qualifying results
Qualifying Results:
{chr(10).join(f1_data['qualifying_results'])}

Step 2: Consider track characteristics
Track: {f1_data['track']}
Track Info: {f1_data['track_characteristics']}
Historical Note: {f1_data['historical_note']}

Step 3: Evaluate weather conditions and their impact
Weather: {f1_data['weather']}

Step 4: Assess each driver's strengths/weaknesses for this specific race
- Consider their qualifying performance
- Think about their historical performance at this track
- Evaluate their current season form
- Consider their team's race pace vs qualifying pace

Step 5: Factor in potential race scenarios
- What could cause position changes?
- Where might overtaking be possible?
- What are the strategic considerations?

Step 6: Make your final prediction
Based on your analysis above, predict the top 3 finishers and provide detailed reasoning for each position.

Please work through each step systematically before making your final prediction.
"""

# Simulated LLM responses for comparison
direct_response = """
Top 3 Prediction:
1. Max Verstappen - Strong qualifying, excellent race pace
2. Charles Leclerc - Good starting position, Ferrari competitive
3. Sergio Perez - Red Bull advantage, consistent performer

Reasoning: Monaco is difficult for overtaking, but Red Bull's race pace advantage should help Verstappen and Perez move up if opportunities arise.
"""

few_shot_response = """
Following the pattern from the examples:

Top 3 Prediction:
1. Max Verstappen - Despite starting P1, his race craft and Red Bull's superior pace
2. Charles Leclerc - Strong P2 start, Ferrari historically good at Monaco
3. Lewis Hamilton - Experience at Monaco, Mercedes improving race pace

Reasoning: Similar to Example 2, Monaco favors qualifying position, but unlike the Ferrari 1-2 example, Verstappen's dominance this season and Red Bull's race pace advantage make him favorite even from pole. Leclerc holds P2 due to track characteristics. Hamilton's Monaco experience gives him edge over Perez for P3.
"""

chain_of_thought_response = """
Step 1: Qualifying Analysis
- Verstappen on pole with strong pace advantage (0.222s over Leclerc)
- Ferrari showing competitive pace with Leclerc P2
- Red Bull 1-3 lockout suggests car performance advantage
- Mercedes improved but still 0.5s+ off pace

Step 2: Track Characteristics  
- Monaco heavily favors qualifying position (65% pole win rate)
- Very limited overtaking opportunities
- Track position more important than pure pace
- Strategy and tire management crucial

Step 3: Weather Impact
- Dry conditions favor consistent performance
- No rain means no major shake-up opportunities
- Stable conditions should maintain grid order more

Step 4: Driver Assessment
- Verstappen: Pole position + current form = huge advantage
- Leclerc: Good qualifier, strong at Monaco historically
- Perez: Consistent but struggles in qualifying, P3 good for him
- Hamilton: Monaco master but Mercedes pace deficit significant
- Russell: Similar to Hamilton but less Monaco experience

Step 5: Race Scenarios
- Safety cars likely in Monaco - key strategic moments
- Pit stop windows limited due to track position importance
- Tire degradation typically low in Monaco
- Driver errors more costly due to barriers

Step 6: Final Prediction
1. Max Verstappen - Pole position + Red Bull pace + current form
2. Charles Leclerc - Strong qualifying position + Ferrari's Monaco competitiveness  
3. Sergio Perez - Red Bull advantage, consistency, good starting position

Detailed reasoning: Monaco's characteristics strongly favor the qualifying order. Verstappen's pole position combined with Red Bull's pace advantage makes him overwhelming favorite. Leclerc's P2 start on a track where overtaking is minimal secures him podium. Perez benefits from Red Bull's race pace to hold off Mercedes challenge.
"""

if __name__ == "__main__":
    print("=== DIRECT PROMPT ===")
    print(direct_prompt())
    print("\n" + "="*50 + "\n")
    
    print("=== FEW-SHOT PROMPT ===") 
    print(few_shot_prompt())
    print("\n" + "="*50 + "\n")
    
    print("=== CHAIN-OF-THOUGHT PROMPT ===")
    print(chain_of_thought_prompt())
