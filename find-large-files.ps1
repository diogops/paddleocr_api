# Busca os 50 maiores arquivos no disco C:
Write-Host "Buscando os maiores arquivos no disco C:..." -ForegroundColor Yellow
Write-Host "Isso pode levar alguns minutos..." -ForegroundColor Yellow
Write-Host ""

$files = Get-ChildItem -Path C:\ -File -Recurse -ErrorAction SilentlyContinue |
    Sort-Object Length -Descending |
    Select-Object -First 50

$fileList = @()
$index = 1

foreach ($file in $files) {
    $sizeGB = [math]::Round($file.Length / 1GB, 2)
    $sizeMB = [math]::Round($file.Length / 1MB, 2)

    $fileList += [PSCustomObject]@{
        Index = $index
        'Tamanho (GB)' = $sizeGB
        'Tamanho (MB)' = $sizeMB
        'Caminho' = $file.FullName
    }
    $index++
}

$fileList | Format-Table -AutoSize

# Exporta para JSON para processamento posterior
$fileList | ConvertTo-Json | Out-File -FilePath "D:\Desenvolvimento\AI\PaddleOCR\large-files-2.json" -Encoding UTF8
Write-Host ""
Write-Host "Lista atualizada salva em large-files-2.json" -ForegroundColor Green
