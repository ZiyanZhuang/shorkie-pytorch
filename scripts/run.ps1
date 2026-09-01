param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$PythonArguments
)

$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

if (-not $PythonArguments) {
  throw 'Pass a Python module or script, for example: .\scripts\run.ps1 -m pytest -q'
}
& python @PythonArguments
exit $LASTEXITCODE
