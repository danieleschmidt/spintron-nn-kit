"""
Advanced Internationalization Framework for SpinTron-NN-Kit.

This module implements comprehensive globalization features:
- Multi-language support for scientific terminology
- Regional adaptation for hardware specifications
- Cultural considerations for research presentation
- Dynamic language switching for global deployment
"""

import json
import os
import locale
import gettext
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SupportedLanguage(Enum):
    """Supported languages for SpinTron-NN-Kit."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh_CN"
    CHINESE_TRADITIONAL = "zh_TW"
    KOREAN = "ko"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"


class RegionalStandard(Enum):
    """Regional standards for hardware specifications."""
    
    INTERNATIONAL = "intl"
    NORTH_AMERICA = "na"
    EUROPE = "eu"
    ASIA_PACIFIC = "apac"
    JAPAN = "jp"
    CHINA = "cn"


@dataclass
class LocalizationConfig:
    """Configuration for localization and internationalization."""
    
    primary_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    regional_standard: RegionalStandard = RegionalStandard.INTERNATIONAL
    
    # Number formatting
    decimal_separator: str = "."
    thousands_separator: str = ","
    
    # Unit preferences
    temperature_unit: str = "celsius"  # celsius, fahrenheit, kelvin
    voltage_unit: str = "volts"        # volts, millivolts
    resistance_unit: str = "ohms"      # ohms, kiloohms, megaohms
    
    # Cultural preferences
    scientific_notation: bool = True
    metric_system: bool = True
    paper_size: str = "A4"  # A4, Letter, Legal
    
    # Display preferences
    date_format: str = "ISO"  # ISO, US, EU, ASIAN
    time_format: str = "24h"  # 12h, 24h


class ScientificTerminologyManager:
    """Manages scientific terminology translation and localization."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.terminology_db = {}
        self.unit_conversions = {}
        self._load_terminology_database()
        self._load_unit_conversions()
    
    def _load_terminology_database(self):
        """Load scientific terminology translations."""
        
        # Core spintronics terminology
        self.terminology_db = {
            "magnetic_tunnel_junction": {
                "en": "Magnetic Tunnel Junction",
                "es": "Unión de Túnel Magnético",
                "fr": "Jonction Tunnel Magnétique",
                "de": "Magnetischer Tunnelübergang",
                "ja": "磁気トンネル接合",
                "zh_CN": "磁隧道结",
                "zh_TW": "磁穿隧接面",
                "ko": "자기터널접합",
                "ru": "Магнитный туннельный переход",
                "pt": "Junção de Túnel Magnético",
                "it": "Giunzione a Tunnel Magnetico",
                "nl": "Magnetische Tunneljunctie"
            },
            "spin_orbit_torque": {
                "en": "Spin-Orbit Torque",
                "es": "Par Espín-Órbita",
                "fr": "Couple Spin-Orbite",
                "de": "Spin-Bahn-Drehmoment",
                "ja": "スピン軌道トルク",
                "zh_CN": "自旋轨道扭矩",
                "zh_TW": "自旋軌道力矩",
                "ko": "스핀궤도토크",
                "ru": "Спин-орбитальный момент",
                "pt": "Torque Spin-Órbita",
                "it": "Coppia Spin-Orbita",
                "nl": "Spin-Baan Koppel"
            },
            "crossbar_array": {
                "en": "Crossbar Array",
                "es": "Matriz de Barras Cruzadas",
                "fr": "Réseau de Barres Croisées",
                "de": "Kreuzschaltmatrix",
                "ja": "クロスバーアレイ",
                "zh_CN": "交叉开关阵列",
                "zh_TW": "交叉桿陣列",
                "ko": "크로스바 배열",
                "ru": "Перекрестная матрица",
                "pt": "Arranjo de Barras Cruzadas",
                "it": "Array a Barre Incrociate",
                "nl": "Kruisbalken Array"
            },
            "neuromorphic_computing": {
                "en": "Neuromorphic Computing",
                "es": "Computación Neuromórfica",
                "fr": "Informatique Neuromorphique",
                "de": "Neuromorphe Datenverarbeitung",
                "ja": "ニューロモルフィックコンピューティング",
                "zh_CN": "神经形态计算",
                "zh_TW": "神經形態計算",
                "ko": "뉴로모픽 컴퓨팅",
                "ru": "Нейроморфные вычисления",
                "pt": "Computação Neuromórfica",
                "it": "Calcolo Neuromorfico",
                "nl": "Neuromorfische Computing"
            },
            "quantization_aware_training": {
                "en": "Quantization-Aware Training",
                "es": "Entrenamiento Consciente de Cuantización",
                "fr": "Entraînement Conscient de la Quantification",
                "de": "Quantisierungsbewusstes Training",
                "ja": "量子化認識トレーニング",
                "zh_CN": "量化感知训练",
                "zh_TW": "量化感知訓練",
                "ko": "양자화 인식 훈련",
                "ru": "Обучение с учетом квантования",
                "pt": "Treinamento Consciente de Quantização",
                "it": "Addestramento Consapevole della Quantizzazione",
                "nl": "Kwantisatiebewuste Training"
            }
        }
    
    def _load_unit_conversions(self):
        """Load unit conversion factors for different regional standards."""
        
        self.unit_conversions = {
            "voltage": {
                "volts_to_millivolts": 1000,
                "volts_to_microvolts": 1e6,
                "millivolts_to_volts": 0.001
            },
            "resistance": {
                "ohms_to_kiloohms": 0.001,
                "ohms_to_megaohms": 1e-6,
                "kiloohms_to_ohms": 1000,
                "megaohms_to_ohms": 1e6
            },
            "energy": {
                "joules_to_picojoules": 1e12,
                "joules_to_femtojoules": 1e15,
                "picojoules_to_joules": 1e-12
            },
            "temperature": {
                "celsius_to_fahrenheit": lambda c: c * 9/5 + 32,
                "fahrenheit_to_celsius": lambda f: (f - 32) * 5/9,
                "celsius_to_kelvin": lambda c: c + 273.15,
                "kelvin_to_celsius": lambda k: k - 273.15
            }
        }
    
    def translate_term(self, term: str, target_language: str = None) -> str:
        """
        Translate scientific term to target language.
        
        Args:
            term: Term to translate
            target_language: Target language code
            
        Returns:
            Translated term or original if not found
        """
        if target_language is None:
            target_language = self.config.primary_language.value
        
        if term in self.terminology_db:
            translations = self.terminology_db[term]
            return translations.get(target_language, translations.get("en", term))
        
        return term
    
    def format_number(self, value: float, precision: int = 3) -> str:
        """Format number according to regional preferences."""
        
        if self.config.scientific_notation and (abs(value) >= 1e6 or abs(value) <= 1e-3):
            formatted = f"{value:.{precision}e}"
        else:
            formatted = f"{value:.{precision}f}"
        
        # Apply regional number formatting
        if self.config.decimal_separator != ".":
            formatted = formatted.replace(".", self.config.decimal_separator)
        
        # Add thousands separator for large numbers
        if self.config.thousands_separator and "e" not in formatted.lower():
            parts = formatted.split(self.config.decimal_separator)
            integer_part = parts[0]
            
            # Add thousands separators
            if len(integer_part) > 3:
                formatted_integer = ""
                for i, digit in enumerate(reversed(integer_part)):
                    if i > 0 and i % 3 == 0:
                        formatted_integer = self.config.thousands_separator + formatted_integer
                    formatted_integer = digit + formatted_integer
                
                if len(parts) > 1:
                    formatted = formatted_integer + self.config.decimal_separator + parts[1]
                else:
                    formatted = formatted_integer
        
        return formatted
    
    def format_unit(self, value: float, unit_type: str, precision: int = 3) -> str:
        """Format value with appropriate unit for regional preferences."""
        
        # Convert to preferred unit
        if unit_type == "voltage":
            if self.config.voltage_unit == "millivolts":
                value *= 1000
                unit = self.translate_term("millivolts")
            else:
                unit = self.translate_term("volts")
                
        elif unit_type == "resistance":
            if self.config.resistance_unit == "kiloohms" and value >= 1000:
                value /= 1000
                unit = self.translate_term("kiloohms")
            elif self.config.resistance_unit == "megaohms" and value >= 1e6:
                value /= 1e6
                unit = self.translate_term("megaohms")
            else:
                unit = self.translate_term("ohms")
                
        elif unit_type == "temperature":
            if self.config.temperature_unit == "fahrenheit":
                value = value * 9/5 + 32
                unit = "°F"
            elif self.config.temperature_unit == "kelvin":
                value += 273.15
                unit = "K"
            else:
                unit = "°C"
        else:
            unit = unit_type
        
        formatted_value = self.format_number(value, precision)
        return f"{formatted_value} {unit}"


class DocumentationLocalizer:
    """Handles localization of documentation and user interfaces."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.messages = {}
        self._load_message_catalogs()
    
    def _load_message_catalogs(self):
        """Load translated message catalogs."""
        
        # Core UI messages
        self.messages = {
            "optimization_started": {
                "en": "Optimization started",
                "es": "Optimización iniciada",
                "fr": "Optimisation démarrée",
                "de": "Optimierung gestartet",
                "ja": "最適化を開始しました",
                "zh_CN": "优化已开始",
                "zh_TW": "優化已開始",
                "ko": "최적화 시작",
                "ru": "Оптимизация начата",
                "pt": "Otimização iniciada",
                "it": "Ottimizzazione avviata",
                "nl": "Optimalisatie gestart"
            },
            "optimization_completed": {
                "en": "Optimization completed successfully",
                "es": "Optimización completada con éxito",
                "fr": "Optimisation terminée avec succès",
                "de": "Optimierung erfolgreich abgeschlossen",
                "ja": "最適化が正常に完了しました",
                "zh_CN": "优化成功完成",
                "zh_TW": "優化成功完成",
                "ko": "최적화 성공적으로 완료",
                "ru": "Оптимизация успешно завершена",
                "pt": "Otimização concluída com sucesso",
                "it": "Ottimizzazione completata con successo",
                "nl": "Optimalisatie succesvol voltooid"
            },
            "energy_consumption": {
                "en": "Energy consumption",
                "es": "Consumo de energía",
                "fr": "Consommation d'énergie",
                "de": "Energieverbrauch",
                "ja": "消費エネルギー",
                "zh_CN": "能耗",
                "zh_TW": "能源消耗",
                "ko": "에너지 소비",
                "ru": "Потребление энергии",
                "pt": "Consumo de energia",
                "it": "Consumo energetico",
                "nl": "Energieverbruik"
            },
            "quantum_advantage": {
                "en": "Quantum advantage",
                "es": "Ventaja cuántica",
                "fr": "Avantage quantique",
                "de": "Quantenvorteil",
                "ja": "量子優位性",
                "zh_CN": "量子优势",
                "zh_TW": "量子優勢",
                "ko": "양자 우위",
                "ru": "Квантовое преимущество",
                "pt": "Vantagem quântica",
                "it": "Vantaggio quantistico",
                "nl": "Quantum voordeel"
            },
            "neural_network_accuracy": {
                "en": "Neural network accuracy",
                "es": "Precisión de la red neuronal",
                "fr": "Précision du réseau de neurones",
                "de": "Neuronale Netzwerkgenauigkeit",
                "ja": "ニューラルネットワークの精度",
                "zh_CN": "神经网络准确性",
                "zh_TW": "神經網路準確性",
                "ko": "신경망 정확도",
                "ru": "Точность нейронной сети",
                "pt": "Precisão da rede neural",
                "it": "Precisione della rete neurale",
                "nl": "Neurale netwerk nauwkeurigheid"
            }
        }
    
    def get_message(self, message_key: str, language: str = None) -> str:
        """Get localized message."""
        if language is None:
            language = self.config.primary_language.value
        
        if message_key in self.messages:
            translations = self.messages[message_key]
            return translations.get(language, translations.get("en", message_key))
        
        return message_key
    
    def format_message(self, message_key: str, **kwargs) -> str:
        """Format localized message with parameters."""
        message = self.get_message(message_key)
        
        try:
            return message.format(**kwargs)
        except (KeyError, ValueError):
            return message


class RegionalStandardsManager:
    """Manages regional standards and hardware specifications."""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.regional_specs = {}
        self._load_regional_specifications()
    
    def _load_regional_specifications(self):
        """Load regional hardware specifications and standards."""
        
        self.regional_specs = {
            "voltage_standards": {
                "na": {"nominal": 1.2, "tolerance": 0.05},      # North America
                "eu": {"nominal": 1.2, "tolerance": 0.03},      # Europe
                "jp": {"nominal": 1.0, "tolerance": 0.02},      # Japan
                "cn": {"nominal": 1.1, "tolerance": 0.04},      # China
                "intl": {"nominal": 1.2, "tolerance": 0.05}     # International
            },
            "frequency_standards": {
                "na": {"base": 50e6, "multipliers": [1, 2, 4, 8]},
                "eu": {"base": 50e6, "multipliers": [1, 2, 4, 8]},
                "jp": {"base": 40e6, "multipliers": [1, 2.5, 5, 10]},
                "cn": {"base": 60e6, "multipliers": [1, 2, 4, 8]},
                "intl": {"base": 50e6, "multipliers": [1, 2, 4, 8]}
            },
            "safety_standards": {
                "na": ["UL", "FCC", "IEEE"],
                "eu": ["CE", "RoHS", "REACH"],
                "jp": ["JIS", "VCCI"],
                "cn": ["CCC", "GB"],
                "intl": ["ISO", "IEC"]
            },
            "environmental_ranges": {
                "na": {"temp_min": -40, "temp_max": 85, "humidity_max": 95},
                "eu": {"temp_min": -40, "temp_max": 85, "humidity_max": 95},
                "jp": {"temp_min": -20, "temp_max": 70, "humidity_max": 85},
                "cn": {"temp_min": -30, "temp_max": 80, "humidity_max": 90},
                "intl": {"temp_min": -40, "temp_max": 85, "humidity_max": 95}
            }
        }
    
    def get_regional_spec(self, spec_type: str, region: str = None) -> Dict[str, Any]:
        """Get regional specification."""
        if region is None:
            region = self.config.regional_standard.value
        
        if spec_type in self.regional_specs:
            specs = self.regional_specs[spec_type]
            return specs.get(region, specs.get("intl", {}))
        
        return {}
    
    def validate_regional_compliance(self, parameters: Dict[str, float]) -> Dict[str, bool]:
        """Validate parameters against regional standards."""
        
        compliance = {}
        region = self.config.regional_standard.value
        
        # Check voltage compliance
        voltage_spec = self.get_regional_spec("voltage_standards")
        if "voltage" in parameters and voltage_spec:
            nominal = voltage_spec["nominal"]
            tolerance = voltage_spec["tolerance"]
            voltage = parameters["voltage"]
            
            min_voltage = nominal * (1 - tolerance)
            max_voltage = nominal * (1 + tolerance)
            compliance["voltage"] = min_voltage <= voltage <= max_voltage
        
        # Check environmental compliance
        env_spec = self.get_regional_spec("environmental_ranges")
        if "temperature" in parameters and env_spec:
            temp = parameters["temperature"]
            compliance["temperature"] = env_spec["temp_min"] <= temp <= env_spec["temp_max"]
        
        return compliance


class GlobalizationManager:
    """Main globalization manager coordinating all localization aspects."""
    
    def __init__(self, config: LocalizationConfig = None):
        if config is None:
            config = LocalizationConfig()
        
        self.config = config
        self.terminology = ScientificTerminologyManager(config)
        self.documentation = DocumentationLocalizer(config)
        self.regional = RegionalStandardsManager(config)
        
        # Set system locale if possible
        self._configure_system_locale()
    
    def _configure_system_locale(self):
        """Configure system locale based on configuration."""
        try:
            language_code = self.config.primary_language.value
            if language_code == "zh_CN":
                locale_code = "zh_CN.UTF-8"
            elif language_code == "zh_TW":
                locale_code = "zh_TW.UTF-8"
            else:
                locale_code = f"{language_code}.UTF-8"
            
            locale.setlocale(locale.LC_ALL, locale_code)
        except locale.Error:
            # Fallback to default locale
            pass
    
    def set_language(self, language: SupportedLanguage):
        """Change the primary language."""
        self.config.primary_language = language
        self._configure_system_locale()
    
    def set_regional_standard(self, standard: RegionalStandard):
        """Change the regional standard."""
        self.config.regional_standard = standard
    
    def translate_scientific_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Translate scientific output data with proper formatting."""
        
        translated_data = {}
        
        for key, value in data.items():
            # Translate key
            translated_key = self.terminology.translate_term(key)
            
            if isinstance(value, (int, float)):
                # Format numerical values
                if key in ["voltage", "resistance", "temperature"]:
                    translated_data[translated_key] = self.terminology.format_unit(value, key)
                else:
                    translated_data[translated_key] = self.terminology.format_number(value)
            
            elif isinstance(value, str):
                # Try to translate string values
                translated_data[translated_key] = self.terminology.translate_term(value)
            
            elif isinstance(value, dict):
                # Recursively translate nested dictionaries
                translated_data[translated_key] = self.translate_scientific_output(value)
            
            else:
                translated_data[translated_key] = value
        
        return translated_data
    
    def generate_localized_report(self, data: Dict[str, Any], title: str) -> str:
        """Generate a localized scientific report."""
        
        language = self.config.primary_language.value
        
        # Report header
        report_lines = [
            f"# {self.terminology.translate_term(title)}",
            "",
            f"**{self.documentation.get_message('optimization_completed')}**",
            ""
        ]
        
        # Translate and format data
        translated_data = self.translate_scientific_output(data)
        
        for key, value in translated_data.items():
            if isinstance(value, dict):
                report_lines.append(f"## {key}")
                for subkey, subvalue in value.items():
                    report_lines.append(f"- {subkey}: {subvalue}")
                report_lines.append("")
            else:
                report_lines.append(f"**{key}**: {value}")
        
        # Regional compliance information
        if isinstance(data, dict) and any(k in data for k in ["voltage", "temperature"]):
            compliance = self.regional.validate_regional_compliance(data)
            
            report_lines.extend([
                "",
                f"## {self.documentation.get_message('regional_compliance', language)}",
            ])
            
            for param, compliant in compliance.items():
                status = "✅" if compliant else "❌"
                param_name = self.terminology.translate_term(param)
                report_lines.append(f"- {param_name}: {status}")
        
        return "\n".join(report_lines)
    
    def create_global_configuration(self) -> Dict[str, Any]:
        """Create a global configuration dictionary."""
        
        return {
            "language": self.config.primary_language.value,
            "regional_standard": self.config.regional_standard.value,
            "formatting": {
                "decimal_separator": self.config.decimal_separator,
                "thousands_separator": self.config.thousands_separator,
                "scientific_notation": self.config.scientific_notation,
                "date_format": self.config.date_format,
                "time_format": self.config.time_format
            },
            "units": {
                "temperature": self.config.temperature_unit,
                "voltage": self.config.voltage_unit,
                "resistance": self.config.resistance_unit,
                "metric_system": self.config.metric_system
            },
            "regional_specs": {
                "voltage_standard": self.regional.get_regional_spec("voltage_standards"),
                "environmental_range": self.regional.get_regional_spec("environmental_ranges"),
                "safety_standards": self.regional.get_regional_spec("safety_standards")
            }
        }


def demonstrate_advanced_globalization():
    """Demonstrate advanced globalization capabilities."""
    
    print("🌍 Advanced Globalization Framework Demo")
    print("=" * 60)
    
    # Test multiple language configurations
    languages_to_test = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.JAPANESE,
        SupportedLanguage.CHINESE_SIMPLIFIED,
        SupportedLanguage.GERMAN,
        SupportedLanguage.SPANISH
    ]
    
    # Test regional standards
    regions_to_test = [
        RegionalStandard.INTERNATIONAL,
        RegionalStandard.JAPAN,
        RegionalStandard.EUROPE,
        RegionalStandard.CHINA
    ]
    
    # Sample scientific data
    sample_data = {
        "magnetic_tunnel_junction": {
            "resistance_high": 25000,
            "resistance_low": 5000,
            "switching_voltage": 0.8,
            "energy_consumption": 12.5e-12,
            "temperature": 85
        },
        "optimization_results": {
            "quantum_advantage": 15.7,
            "neural_network_accuracy": 0.947,
            "optimization_time": 23.4
        }
    }
    
    for lang in languages_to_test[:3]:  # Test first 3 languages
        print(f"\n🗣️  Testing {lang.value.upper()} localization:")
        
        for region in regions_to_test[:2]:  # Test first 2 regions
            config = LocalizationConfig(
                primary_language=lang,
                regional_standard=region
            )
            
            globalizer = GlobalizationManager(config)
            
            print(f"   📍 Region: {region.value}")
            
            # Test terminology translation
            mtj_term = globalizer.terminology.translate_term("magnetic_tunnel_junction")
            print(f"      MTJ: {mtj_term}")
            
            # Test number formatting
            voltage = globalizer.terminology.format_unit(0.8, "voltage")
            energy = globalizer.terminology.format_number(12.5e-12)
            print(f"      Voltage: {voltage}")
            print(f"      Energy: {energy}")
            
            # Test regional compliance
            compliance = globalizer.regional.validate_regional_compliance(sample_data["magnetic_tunnel_junction"])
            compliance_status = "✅" if all(compliance.values()) else "⚠️"
            print(f"      Compliance: {compliance_status}")
    
    # Generate comprehensive localized report
    print(f"\n📄 Sample Localized Reports:")
    
    # English report
    en_config = LocalizationConfig(primary_language=SupportedLanguage.ENGLISH)
    en_globalizer = GlobalizationManager(en_config)
    en_report = en_globalizer.generate_localized_report(sample_data, "optimization_results")
    
    # Japanese report
    ja_config = LocalizationConfig(primary_language=SupportedLanguage.JAPANESE)
    ja_globalizer = GlobalizationManager(ja_config)
    ja_report = ja_globalizer.generate_localized_report(sample_data, "optimization_results")
    
    print("\n🇺🇸 English Report:")
    print("-" * 40)
    print(en_report[:200] + "..." if len(en_report) > 200 else en_report)
    
    print("\n🇯🇵 Japanese Report:")
    print("-" * 40)
    print(ja_report[:200] + "..." if len(ja_report) > 200 else ja_report)
    
    # Global configuration summary
    print(f"\n⚙️  Global Configuration Summary:")
    print("=" * 40)
    
    global_config = en_globalizer.create_global_configuration()
    for section, values in global_config.items():
        print(f"{section.title()}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")
        print()
    
    print("🎯 Advanced Globalization Features Demonstrated:")
    print("✓ Multi-language scientific terminology translation")
    print("✓ Regional hardware standards compliance")
    print("✓ Cultural number and unit formatting")
    print("✓ Dynamic language switching")
    print("✓ Automated localized report generation")
    print("✓ Comprehensive globalization configuration")
    
    return globalizer


if __name__ == "__main__":
    # Run advanced globalization demonstration
    globalizer = demonstrate_advanced_globalization()
    print("\n🌟 Advanced globalization framework demonstration completed!")