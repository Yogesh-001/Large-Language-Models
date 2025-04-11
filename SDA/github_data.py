import os
import json
from time import sleep
from dotenv import load_dotenv
from github import Github, GithubException

load_dotenv()

def safe_get_issues(repo, max_issues=50):
    """Safely get issues with error handling and rate limit awareness"""
    issues = []
    try:
        for issue in repo.get_issues(state="closed", labels=["bug"]):
            if len(issues) >= max_issues:
                break
            issues.append(issue)
            # Basic rate limit handling
            if len(issues) % 10 == 0:
                sleep(1)
    except GithubException as e:
        print(f"Error getting issues for {repo.full_name}: {e}")
    return issues

def process_repository(repo, max_issues=50):
    """Process a single repository and return collected data"""
    repo_data = []
    print(f"\nProcessing {repo.full_name}...")
    
    issues = safe_get_issues(repo, max_issues)
    print(f"Found {len(issues)} bug issues to process")
    
    for i, issue in enumerate(issues, 1):
        print(f"  Processing issue {i}/{len(issues)}: {issue.title[:50]}...", end="\r")
        
        if issue.pull_request:
            try:
                pr = issue.as_pull_request()
                files = pr.get_files()
                
                for f in files:
                    if f.patch and f.filename.endswith(".py"):
                        repo_data.append({
                            "repo": repo.full_name,
                            "issue_number": issue.number,
                            "issue_title": issue.title,
                            "issue_body": issue.body,
                            "patch": f.patch,
                            "file_name": f.filename,
                            "pr_number": pr.number,
                            "pr_url": pr.html_url
                        })
                # Be gentle with the API
                sleep(0.5)
                
            except GithubException as e:
                print(f"\nError processing PR {issue.number}: {e}")
                continue
    
    print(f"\nProcessed {len(repo_data)} valid patches from {repo.full_name}")
    return repo_data

def main():
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN not found in environment variables")
    
    g = Github(GITHUB_TOKEN)
    
    repos_to_process = [
        "python/cpython",
        "pallets/flask",
        "scikit-learn/scikit-learn",
        "TheAlgorithms/Python",
        "Python-World/python-mini-projects"
    ]
    
    all_data = []
    
    try:
        for repo_name in repos_to_process:
            try:
                repo = g.get_repo(repo_name)
                repo_data = process_repository(repo)
                all_data.extend(repo_data)

                with open("github_issues_with_fixes.json", "w") as f:
                    json.dump(all_data, f, indent=2)
                    
            except GithubException as e:
                print(f"Error processing repository {repo_name}: {e}")
                continue
                
    finally:
        with open("github_issues_with_fixes.json", "w") as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\nDone! Collected {len(all_data)} patches total")
        print(f"Results saved to github_issues_with_fixes.json")

if __name__ == "__main__":
    main()