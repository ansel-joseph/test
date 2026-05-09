import subprocess
import webbrowser

apps = {
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "vscode": r"C:\Users\Ansel\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "calculator": "calc.exe",
    "notepad": "notepad.exe"
}


def execute_command(command):
    command = command.lower()

    if "open youtube" in command:
        webbrowser.open("https://youtube.com")
        return "Opening YouTube. Because apparently silence is unbearable."

    for app in apps:
        if f"open {app}" in command:
            subprocess.Popen(apps[app])
            return f"Opening {app}. Humanity survives another day."

    return None