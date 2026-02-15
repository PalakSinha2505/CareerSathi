def calculate_overall_score(analysis_results: list) -> dict:
    categories = ["clarity", "communication", "confidence", "structure", "english"]

    default_response = {
        "average_scores": {cat: 0.0 for cat in categories},
        "overall_score": 0.0,
        "performance_level": "Insufficient Data",
        "summary": "Not enough valid responses were available to generate a reliable score."
    }

    if not analysis_results:
        return default_response

    totals = {cat: 0 for cat in categories}
    valid_entries = 0

    for analysis in analysis_results:
        if not isinstance(analysis, dict):
            continue

        scores = analysis.get("scores")
        if not isinstance(scores, dict):
            continue

        for cat in categories:
            totals[cat] += scores.get(cat, 0)

        valid_entries += 1

    if valid_entries == 0:
        return default_response

    average_scores = {
        cat: round(totals[cat] / valid_entries, 2)
        for cat in categories
    }

    overall_score = round(
        sum(average_scores.values()) / len(categories), 2
    )

    weakest = min(average_scores, key=average_scores.get)
    strongest = max(average_scores, key=average_scores.get)

    if overall_score >= 8.5:
        performance_level = "Excellent"
    elif overall_score >= 7:
        performance_level = "Good"
    elif overall_score >= 5:
        performance_level = "Average"
    else:
        performance_level = "Needs Improvement"

    summary = (
        f"Overall performance was {performance_level.lower()}. "
        f"Strongest area observed was {strongest}, while {weakest} "
        f"consistently reduced the impact of responses."
    )

    return {
        "average_scores": average_scores,
        "overall_score": overall_score,
        "performance_level": performance_level,
        "summary": summary
    }
