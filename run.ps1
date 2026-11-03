Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Predict the Genome Pattern: DNA Sequence Classifier" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Streamlit Web Application..." -ForegroundColor Yellow

$pythonPath = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
if (Test-Path $pythonPath) {
    & $pythonPath -m streamlit run app.py
} else {
    python -m streamlit run app.py
}
