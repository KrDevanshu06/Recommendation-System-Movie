import os
import subprocess
import sys

if __name__ == "__main__":
    # Définir le chemin du projet
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Construire le chemin vers app.py
    app_path = os.path.join(project_root, "streamlit_app", "app.py")
    
    # Lancer Streamlit avec subprocess
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
    except FileNotFoundError:
        print("Error: Streamlit command not found. Make sure Streamlit is installed correctly.")
