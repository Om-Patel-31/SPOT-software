
# triangulated_face_realtime

This is a real-time face recognition project I've been working on. It uses MediaPipe for tracking faces and Gemini to help identify people. There are two versions: a normal one and another that can slowly build the face library while it's running.

## Running it

You'll need Python, the packages in `requirements.txt`, and a Gemini API key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then run

```powershell
python triangulated_face_realtime.py
```

or

```powershell
python triangulated_face_realtime_autotrain.py
```

if you want the automatic training version.

## Controls

- q - quit
- e - start adding someone
- space - capture a picture
- s - save
- c - cancel
- g - ask Gemini
- a - accept the suggestion

The auto-training version also has a few extra controls for approving or rejecting suggestions.

If you want to see what it's doing while it's running, start

```powershell
python gemini_accuracy_dashboard.py
```

and open

```
http://localhost:5000
```

in your browser.

Everything gets logged into the `models` folder, so if something goes wrong you can always look back at what happened.

## Windows build

```powershell
python build_windows_release.py
```

This creates a Windows executable and a ZIP file inside the `dist` folder.

If Windows says it doesn't recognize the application, click **More info** and then **Run anyway**.

You'll also need to set a Gemini API key if you want the Gemini features to work.
