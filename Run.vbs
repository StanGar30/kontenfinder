Option Explicit

Dim shell, fso, baseDir, batchPath, quote, command
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
batchPath = fso.BuildPath(baseDir, fso.GetBaseName(WScript.ScriptFullName) & ".bat")
quote = Chr(34)

shell.CurrentDirectory = baseDir
command = quote & shell.ExpandEnvironmentStrings("%ComSpec%") & quote & _
    " /d /c " & quote & quote & batchPath & quote & quote
shell.Run command, 0, False
