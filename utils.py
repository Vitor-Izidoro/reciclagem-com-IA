# Utilitários: funções de logging simples usadas por vários módulos do projeto
"""
Módulo de utilitários para logging e funções auxiliares
"""
import datetime
from typing import Any

def get_timestamp() -> str:
    """Retorna timestamp formatado"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_info(message: Any) -> None:
    """Log de informação"""
    timestamp = get_timestamp()
    print(f"[INFO] {timestamp} - {message}")

def log_success(message: Any) -> None:
    """Log de sucesso"""
    timestamp = get_timestamp()
    print(f"[SUCCESS] {timestamp} - {message}")

def log_error(message: Any) -> None:
    """Log de erro"""
    timestamp = get_timestamp()
    print(f"[ERROR] {timestamp} - {message}")

def log_warning(message: Any) -> None:
    """Log de aviso"""
    timestamp = get_timestamp()
    print(f"[WARNING] {timestamp} - {message}")

def log_debug(message: Any) -> None:
    """Log de debug"""
    timestamp = get_timestamp()
    print(f"[DEBUG] {timestamp} - {message}")
