"""
Internationalization (i18n) support for SpinTron-NN-Kit.

This module provides comprehensive internationalization capabilities:
- Multi-language support (en, es, fr, de, ja, zh)
- Dynamic locale switching
- Message formatting and pluralization
- Cultural adaptation for neural network metrics
- Localized documentation generation
"""

import json
import os
import locale
import gettext
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SupportedLocale(Enum):
    """Supported locales for SpinTron-NN-Kit."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"


@dataclass
class LocalizationConfig:
    """Configuration for localization system."""
    default_locale: str = "en"
    fallback_locale: str = "en"
    supported_locales: List[str] = None
    auto_detect_locale: bool = True
    locale_directory: str = "locales"
    
    def __post_init__(self):
        if self.supported_locales is None:
            self.supported_locales = [locale.value for locale in SupportedLocale]


class TranslationEngine:
    """Core translation engine with advanced formatting."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.translations = {}
        self.plural_rules = {}
        self.current_locale = config.default_locale
        
        # Load translations
        self._load_translations()
        
        # Load plural rules
        self._load_plural_rules()
    
    def _load_translations(self):
        """Load translation dictionaries for all supported locales."""
        # Default English translations
        self.translations = {
            "en": {
                # System messages
                "system.startup": "SpinTron-NN-Kit starting up...",
                "system.shutdown": "SpinTron-NN-Kit shutting down...",
                "system.ready": "System ready for inference",
                "system.error": "System error occurred",
                
                # Performance messages
                "perf.crossbar_ops": "Crossbar operations: {count} ops/sec",
                "perf.energy_consumption": "Energy consumption: {energy:.2f} pJ/MAC",
                "perf.throughput": "Throughput: {rate:.0f} inferences/sec",
                "perf.latency": "Latency: {time:.2f} ms",
                
                # Model messages
                "model.loading": "Loading model: {model_name}",
                "model.loaded": "Model loaded successfully",
                "model.conversion": "Converting PyTorch model to spintronic format",
                "model.optimizing": "Optimizing model for {device} deployment",
                
                # Error messages
                "error.validation": "Input validation failed: {reason}",
                "error.hardware": "Hardware error in {component}: {message}",
                "error.network": "Network connectivity issue",
                "error.memory": "Insufficient memory for operation",
                
                # Security messages
                "security.access_denied": "Access denied for operation",
                "security.audit_log": "Security event logged",
                "security.compliance_check": "Compliance validation passed",
                
                # Units and metrics
                "unit.picojoule": "pJ",
                "unit.microsecond": "μs", 
                "unit.millisecond": "ms",
                "unit.operations_per_second": "ops/sec",
                "unit.inferences_per_second": "inf/sec",
                
                # Status messages
                "status.healthy": "Healthy",
                "status.degraded": "Degraded",
                "status.critical": "Critical",
                "status.offline": "Offline"
            },
            
            "es": {
                # Sistema
                "system.startup": "Iniciando SpinTron-NN-Kit...",
                "system.shutdown": "Apagando SpinTron-NN-Kit...",
                "system.ready": "Sistema listo para inferencia",
                "system.error": "Ocurrió un error del sistema",
                
                # Rendimiento
                "perf.crossbar_ops": "Operaciones de crossbar: {count} ops/seg",
                "perf.energy_consumption": "Consumo de energía: {energy:.2f} pJ/MAC",
                "perf.throughput": "Rendimiento: {rate:.0f} inferencias/seg",
                "perf.latency": "Latencia: {time:.2f} ms",
                
                # Modelo
                "model.loading": "Cargando modelo: {model_name}",
                "model.loaded": "Modelo cargado exitosamente",
                "model.conversion": "Convirtiendo modelo PyTorch a formato espintrónico",
                "model.optimizing": "Optimizando modelo para implementación en {device}",
                
                # Errores
                "error.validation": "Falló la validación de entrada: {reason}",
                "error.hardware": "Error de hardware en {component}: {message}",
                "error.network": "Problema de conectividad de red",
                "error.memory": "Memoria insuficiente para la operación",
                
                # Seguridad
                "security.access_denied": "Acceso denegado para la operación",
                "security.audit_log": "Evento de seguridad registrado",
                "security.compliance_check": "Validación de cumplimiento aprobada",
                
                # Estados
                "status.healthy": "Saludable",
                "status.degraded": "Degradado",
                "status.critical": "Crítico",
                "status.offline": "Desconectado"
            },
            
            "fr": {
                # Système
                "system.startup": "Démarrage de SpinTron-NN-Kit...",
                "system.shutdown": "Arrêt de SpinTron-NN-Kit...",
                "system.ready": "Système prêt pour l'inférence",
                "system.error": "Erreur système survenue",
                
                # Performance
                "perf.crossbar_ops": "Opérations crossbar: {count} ops/sec",
                "perf.energy_consumption": "Consommation d'énergie: {energy:.2f} pJ/MAC",
                "perf.throughput": "Débit: {rate:.0f} inférences/sec",
                "perf.latency": "Latence: {time:.2f} ms",
                
                # Modèle
                "model.loading": "Chargement du modèle: {model_name}",
                "model.loaded": "Modèle chargé avec succès",
                "model.conversion": "Conversion du modèle PyTorch vers format spintronique",
                "model.optimizing": "Optimisation du modèle pour déploiement {device}",
                
                # Erreurs
                "error.validation": "Échec de validation d'entrée: {reason}",
                "error.hardware": "Erreur matérielle dans {component}: {message}",
                "error.network": "Problème de connectivité réseau",
                "error.memory": "Mémoire insuffisante pour l'opération",
                
                # Sécurité
                "security.access_denied": "Accès refusé pour l'opération",
                "security.audit_log": "Événement de sécurité enregistré",
                "security.compliance_check": "Validation de conformité réussie",
                
                # États
                "status.healthy": "Sain",
                "status.degraded": "Dégradé", 
                "status.critical": "Critique",
                "status.offline": "Hors ligne"
            },
            
            "de": {
                # System
                "system.startup": "SpinTron-NN-Kit startet...",
                "system.shutdown": "SpinTron-NN-Kit wird heruntergefahren...",
                "system.ready": "System bereit für Inferenz",
                "system.error": "Systemfehler aufgetreten",
                
                # Leistung
                "perf.crossbar_ops": "Crossbar-Operationen: {count} ops/Sek",
                "perf.energy_consumption": "Energieverbrauch: {energy:.2f} pJ/MAC",
                "perf.throughput": "Durchsatz: {rate:.0f} Inferenzen/Sek",
                "perf.latency": "Latenz: {time:.2f} ms",
                
                # Modell
                "model.loading": "Modell wird geladen: {model_name}",
                "model.loaded": "Modell erfolgreich geladen",
                "model.conversion": "PyTorch-Modell wird zu spintronischem Format konvertiert",
                "model.optimizing": "Modell wird für {device}-Bereitstellung optimiert",
                
                # Fehler
                "error.validation": "Eingabevalidierung fehlgeschlagen: {reason}",
                "error.hardware": "Hardware-Fehler in {component}: {message}",
                "error.network": "Netzwerkverbindungsproblem",
                "error.memory": "Unzureichender Speicher für Operation",
                
                # Sicherheit
                "security.access_denied": "Zugriff für Operation verweigert",
                "security.audit_log": "Sicherheitsereignis protokolliert",
                "security.compliance_check": "Compliance-Validierung bestanden",
                
                # Status
                "status.healthy": "Gesund",
                "status.degraded": "Degradiert",
                "status.critical": "Kritisch", 
                "status.offline": "Offline"
            },
            
            "ja": {
                # システム
                "system.startup": "SpinTron-NN-Kitを起動中...",
                "system.shutdown": "SpinTron-NN-Kitをシャットダウン中...",
                "system.ready": "推論システム準備完了",
                "system.error": "システムエラーが発生しました",
                
                # パフォーマンス
                "perf.crossbar_ops": "クロスバー操作: {count} ops/秒",
                "perf.energy_consumption": "エネルギー消費: {energy:.2f} pJ/MAC",
                "perf.throughput": "スループット: {rate:.0f} 推論/秒",
                "perf.latency": "レイテンシ: {time:.2f} ms",
                
                # モデル
                "model.loading": "モデルをロード中: {model_name}",
                "model.loaded": "モデルが正常にロードされました",
                "model.conversion": "PyTorchモデルをスピントロニクス形式に変換中",
                "model.optimizing": "{device}展開用にモデルを最適化中",
                
                # エラー
                "error.validation": "入力検証に失敗: {reason}",
                "error.hardware": "{component}でハードウェアエラー: {message}",
                "error.network": "ネットワーク接続の問題",
                "error.memory": "操作に十分なメモリがありません",
                
                # セキュリティ
                "security.access_denied": "操作のアクセスが拒否されました",
                "security.audit_log": "セキュリティイベントがログに記録されました",
                "security.compliance_check": "コンプライアンス検証が通過しました",
                
                # ステータス
                "status.healthy": "正常",
                "status.degraded": "劣化",
                "status.critical": "重要",
                "status.offline": "オフライン"
            },
            
            "zh": {
                # 系统
                "system.startup": "正在启动SpinTron-NN-Kit...",
                "system.shutdown": "正在关闭SpinTron-NN-Kit...",
                "system.ready": "系统已准备好进行推理",
                "system.error": "发生系统错误",
                
                # 性能
                "perf.crossbar_ops": "交叉开关操作: {count} ops/秒",
                "perf.energy_consumption": "能耗: {energy:.2f} pJ/MAC",
                "perf.throughput": "吞吐量: {rate:.0f} 推理/秒",
                "perf.latency": "延迟: {time:.2f} ms",
                
                # 模型
                "model.loading": "正在加载模型: {model_name}",
                "model.loaded": "模型加载成功",
                "model.conversion": "正在将PyTorch模型转换为自旋电子格式",
                "model.optimizing": "正在为{device}部署优化模型",
                
                # 错误
                "error.validation": "输入验证失败: {reason}",
                "error.hardware": "{component}硬件错误: {message}",
                "error.network": "网络连接问题",
                "error.memory": "操作内存不足",
                
                # 安全
                "security.access_denied": "操作访问被拒绝",
                "security.audit_log": "安全事件已记录",
                "security.compliance_check": "合规验证通过",
                
                # 状态
                "status.healthy": "健康",
                "status.degraded": "降级",
                "status.critical": "紧急",
                "status.offline": "离线"
            }
        }
    
    def _load_plural_rules(self):
        """Load pluralization rules for different languages."""
        self.plural_rules = {
            "en": lambda n: 0 if n == 1 else 1,
            "es": lambda n: 0 if n == 1 else 1,
            "fr": lambda n: 0 if n == 0 or n == 1 else 1,
            "de": lambda n: 0 if n == 1 else 1,
            "ja": lambda n: 0,  # Japanese doesn't have plural forms
            "zh": lambda n: 0   # Chinese doesn't have plural forms
        }
    
    def set_locale(self, locale: str) -> bool:
        """Set current locale.
        
        Args:
            locale: Locale code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if locale was set successfully
        """
        if locale in self.config.supported_locales:
            self.current_locale = locale
            return True
        return False
    
    def get_message(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Get localized message.
        
        Args:
            key: Message key
            locale: Specific locale (uses current if None)
            **kwargs: Format parameters
            
        Returns:
            Localized message
        """
        target_locale = locale or self.current_locale
        
        # Try target locale
        if (target_locale in self.translations and 
            key in self.translations[target_locale]):
            message = self.translations[target_locale][key]
        # Fallback to default locale
        elif (self.config.fallback_locale in self.translations and
              key in self.translations[self.config.fallback_locale]):
            message = self.translations[self.config.fallback_locale][key]
        # Last resort: return key
        else:
            return key
        
        # Format message with parameters
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError):
            return message
    
    def format_number(self, number: Union[int, float], locale: Optional[str] = None) -> str:
        """Format number according to locale conventions.
        
        Args:
            number: Number to format
            locale: Target locale
            
        Returns:
            Formatted number string
        """
        target_locale = locale or self.current_locale
        
        # Locale-specific number formatting
        if target_locale in ["en"]:
            return f"{number:,}"
        elif target_locale in ["es", "fr"]:
            return f"{number:,}".replace(",", " ")
        elif target_locale in ["de"]:
            return f"{number:,}".replace(",", ".")
        elif target_locale in ["ja", "zh"]:
            return str(number)
        else:
            return str(number)
    
    def format_energy(self, energy_pj: float, locale: Optional[str] = None) -> str:
        """Format energy value with appropriate unit.
        
        Args:
            energy_pj: Energy in picojoules
            locale: Target locale
            
        Returns:
            Formatted energy string
        """
        unit_key = "unit.picojoule"
        unit = self.get_message(unit_key, locale)
        formatted_number = self.format_number(energy_pj, locale)
        
        return f"{formatted_number} {unit}"
    
    def format_time(self, time_ms: float, locale: Optional[str] = None) -> str:
        """Format time value with appropriate unit.
        
        Args:
            time_ms: Time in milliseconds
            locale: Target locale
            
        Returns:
            Formatted time string
        """
        if time_ms < 1.0:
            # Use microseconds
            time_us = time_ms * 1000
            unit_key = "unit.microsecond"
            unit = self.get_message(unit_key, locale)
            formatted_number = self.format_number(time_us, locale)
        else:
            # Use milliseconds
            unit_key = "unit.millisecond"
            unit = self.get_message(unit_key, locale)
            formatted_number = self.format_number(time_ms, locale)
        
        return f"{formatted_number} {unit}"


class InternationalizationManager:
    """Main i18n manager for SpinTron-NN-Kit."""
    
    def __init__(self, config: Optional[LocalizationConfig] = None):
        self.config = config or LocalizationConfig()
        self.translation_engine = TranslationEngine(self.config)
        
        # Auto-detect locale if enabled
        if self.config.auto_detect_locale:
            self._auto_detect_locale()
    
    def _auto_detect_locale(self):
        """Auto-detect system locale."""
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                # Extract language code (e.g., 'en_US' -> 'en')
                lang_code = system_locale.split('_')[0]
                if lang_code in self.config.supported_locales:
                    self.translation_engine.set_locale(lang_code)
        except Exception:
            # Use default locale on failure
            pass
    
    def set_locale(self, locale_code: str) -> bool:
        """Set active locale.
        
        Args:
            locale_code: Locale code
            
        Returns:
            True if successful
        """
        return self.translation_engine.set_locale(locale_code)
    
    def get_current_locale(self) -> str:
        """Get current active locale."""
        return self.translation_engine.current_locale
    
    def get_supported_locales(self) -> List[str]:
        """Get list of supported locales."""
        return self.config.supported_locales.copy()
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate message key to current locale.
        
        Args:
            key: Message key
            **kwargs: Format parameters
            
        Returns:
            Translated message
        """
        return self.translation_engine.get_message(key, **kwargs)
    
    def format_performance_metric(self, metric_type: str, value: float) -> str:
        """Format performance metric with localization.
        
        Args:
            metric_type: Type of metric ('energy', 'latency', 'throughput', etc.)
            value: Metric value
            
        Returns:
            Formatted metric string
        """
        if metric_type == "energy":
            return self.translation_engine.format_energy(value)
        elif metric_type == "latency":
            return self.translation_engine.format_time(value)
        elif metric_type in ["throughput", "ops_per_sec"]:
            formatted_number = self.translation_engine.format_number(value)
            unit = self.translate("unit.operations_per_second")
            return f"{formatted_number} {unit}"
        else:
            return self.translation_engine.format_number(value)
    
    def get_localized_status(self, status: str) -> str:
        """Get localized status message.
        
        Args:
            status: Status key ('healthy', 'degraded', etc.)
            
        Returns:
            Localized status
        """
        return self.translate(f"status.{status}")
    
    def log_system_message(self, message_key: str, **kwargs) -> str:
        """Log system message with localization.
        
        Args:
            message_key: System message key
            **kwargs: Format parameters
            
        Returns:
            Localized message
        """
        return self.translate(f"system.{message_key}", **kwargs)


# Global i18n instance
_global_i18n_manager = None

def get_i18n_manager() -> InternationalizationManager:
    """Get global i18n manager instance."""
    global _global_i18n_manager
    
    if _global_i18n_manager is None:
        _global_i18n_manager = InternationalizationManager()
    
    return _global_i18n_manager


def get_localized_message(key: str, **kwargs) -> str:
    """Get localized message using global i18n manager.
    
    Args:
        key: Message key
        **kwargs: Format parameters
        
    Returns:
        Localized message
    """
    return get_i18n_manager().translate(key, **kwargs)


def set_global_locale(locale_code: str) -> bool:
    """Set global locale.
    
    Args:
        locale_code: Locale code
        
    Returns:
        True if successful
    """
    return get_i18n_manager().set_locale(locale_code)


def format_metric(metric_type: str, value: float) -> str:
    """Format metric with global i18n manager.
    
    Args:
        metric_type: Metric type
        value: Value
        
    Returns:
        Formatted metric
    """
    return get_i18n_manager().format_performance_metric(metric_type, value)