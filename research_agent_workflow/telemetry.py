import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import os
from pathlib import Path


class EventType(Enum):
    """Types of events to track"""
    USER_INPUT = "user_input"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    VALIDATION = "validation"
    ERROR = "error"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    AGENT_EXECUTION = "agent_execution"


class LogLevel(Enum):
    """Log levels for telemetry"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TelemetryJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Enums and other special types"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


@dataclass
class TelemetryEvent:
    """Structure for telemetry events"""
    event_id: str
    session_id: str
    timestamp: str
    event_type: EventType
    log_level: LogLevel
    component: str
    message: str
    data: Dict[str, Any]
    execution_time_ms: Optional[float] = None
    user_id: Optional[str] = None
    patient_id: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class TelemetryCollector:
    """Main telemetry collection class"""
    
    def __init__(self, 
                 session_id: Optional[str] = None,
                 log_to_file: bool = True,
                 log_to_console: bool = True,
                 log_directory: str = "logs",
                 max_log_files: int = 10):
        
        self.session_id = session_id or str(uuid.uuid4())
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_directory = Path(log_directory)
        self.max_log_files = max_log_files
        
        # Create logs directory if it doesn't exist
        if self.log_to_file:
            self.log_directory.mkdir(exist_ok=True)
            self._setup_file_logging()
        
        if self.log_to_console:
            self._setup_console_logging()
        
        # In-memory storage for session analytics
        self.events: List[TelemetryEvent] = []
        self.session_start_time = time.time()
        
        # Track session metrics
        self.metrics = {
            "total_events": 0,
            "error_count": 0,
            "llm_requests": 0,
            "validation_attempts": 0,
            "successful_workflows": 0,
            "failed_workflows": 0
        }
        
        self.log_event(
            event_type=EventType.WORKFLOW_START,
            component="TelemetryCollector",
            message=f"Telemetry session started: {self.session_id}",
            data={"session_id": self.session_id}
        )
    
    def _setup_file_logging(self):
        """Setup file-based logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = self.log_directory / f"telemetry_{timestamp}_{self.session_id[:8]}.jsonl"
        
        self.file_logger = logging.getLogger(f"telemetry_file_{self.session_id}")
        self.file_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        self.file_logger.handlers.clear()
        
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.file_logger.addHandler(file_handler)
        self.file_logger.propagate = False
    
    def _setup_console_logging(self):
        """Setup console logging"""
        self.console_logger = logging.getLogger(f"telemetry_console_{self.session_id}")
        self.console_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.console_logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        self.console_logger.addHandler(console_handler)
        self.console_logger.propagate = False
    
    def log_event(self,
                  event_type: EventType,
                  component: str,
                  message: str,
                  data: Dict[str, Any] = None,
                  log_level: LogLevel = LogLevel.INFO,
                  execution_time_ms: Optional[float] = None,
                  user_id: Optional[str] = None,
                  patient_id: Optional[str] = None,
                  error_details: Optional[Dict[str, Any]] = None):
        """Log a telemetry event"""
        
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            log_level=log_level,
            component=component,
            message=message,
            data=data or {},
            execution_time_ms=execution_time_ms,
            user_id=user_id,
            patient_id=patient_id,
            error_details=error_details
        )
        
        # Add to in-memory storage
        self.events.append(event)
        
        # Update metrics
        self._update_metrics(event)
        
        # Log to file with custom JSON encoder
        if self.log_to_file:
            self.file_logger.info(json.dumps(asdict(event), cls=TelemetryJSONEncoder, indent=None))
        
        # Log to console
        if self.log_to_console:
            console_message = f"[{event_type.value.upper()}] {component}: {message}"
            if execution_time_ms:
                console_message += f" (took {execution_time_ms:.2f}ms)"
            
            if log_level == LogLevel.ERROR or log_level == LogLevel.CRITICAL:
                self.console_logger.error(console_message)
            elif log_level == LogLevel.WARNING:
                self.console_logger.warning(console_message)
            else:
                self.console_logger.info(console_message)
    
    def _update_metrics(self, event: TelemetryEvent):
        """Update session metrics based on event"""
        self.metrics["total_events"] += 1
        
        if event.event_type == EventType.ERROR:
            self.metrics["error_count"] += 1
        elif event.event_type == EventType.LLM_REQUEST:
            self.metrics["llm_requests"] += 1
        elif event.event_type == EventType.VALIDATION:
            self.metrics["validation_attempts"] += 1
        elif event.event_type == EventType.WORKFLOW_END:
            if event.log_level == LogLevel.INFO:
                self.metrics["successful_workflows"] += 1
            else:
                self.metrics["failed_workflows"] += 1
    
    def log_user_input(self, patient_data: Dict[str, Any], patient_id: str):
        """Log user input data"""
        self.log_event(
            event_type=EventType.USER_INPUT,
            component="UserInput",
            message=f"Patient data received for {patient_id}",
            data={
                "patient_id": patient_id,
                "age": patient_data.get("age"),
                "gender": patient_data.get("gender"),
                "condition_length": len(patient_data.get("medical_condition", "")),
                "history_items": len(patient_data.get("medical_history", []))
            },
            patient_id=patient_id
        )
    
    def log_llm_request(self, agent_name: str, model: str, prompt_length: int, patient_id: str):
        """Log LLM API request"""
        start_time = time.time()
        
        self.log_event(
            event_type=EventType.LLM_REQUEST,
            component=agent_name,
            message=f"LLM request initiated to {model}",
            data={
                "model": model,
                "prompt_length": prompt_length,
                "request_start_time": start_time
            },
            patient_id=patient_id
        )
        
        return start_time
    
    def log_llm_response(self, agent_name: str, start_time: float, response_data: Dict[str, Any], 
                        patient_id: str, success: bool = True):
        """Log LLM API response"""
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        self.log_event(
            event_type=EventType.LLM_RESPONSE,
            component=agent_name,
            message=f"LLM response {'successful' if success else 'failed'}",
            data={
                "success": success,
                "response_length": len(str(response_data)),
                "contains_error": "error" in response_data,
                "response_keys": list(response_data.keys()) if isinstance(response_data, dict) else []
            },
            execution_time_ms=execution_time,
            patient_id=patient_id,
            log_level=LogLevel.INFO if success else LogLevel.ERROR
        )
    
    def log_validation_result(self, validation_data: Dict[str, Any], patient_id: str):
        """Log validation results"""
        valid_count = sum(1 for v in validation_data.values() if v == "valid")
        total_count = len(validation_data)
        
        self.log_event(
            event_type=EventType.VALIDATION,
            component="ValidationAgent",
            message=f"Validation completed: {valid_count}/{total_count} items valid",
            data={
                "validation_results": validation_data,
                "valid_count": valid_count,
                "total_count": total_count,
                "validation_rate": valid_count / total_count if total_count > 0 else 0
            },
            patient_id=patient_id
        )
    
    def log_error(self, component: str, error_message: str, error_details: Dict[str, Any] = None,
                  patient_id: Optional[str] = None):
        """Log error events"""
        self.log_event(
            event_type=EventType.ERROR,
            component=component,
            message=error_message,
            data=error_details or {},
            log_level=LogLevel.ERROR,
            patient_id=patient_id,
            error_details=error_details
        )
    
    def log_agent_execution(self, agent_name: str, task_name: str, start_time: float, 
                           success: bool, patient_id: str):
        """Log agent execution metrics"""
        execution_time = (time.time() - start_time) * 1000
        
        self.log_event(
            event_type=EventType.AGENT_EXECUTION,
            component=agent_name,
            message=f"Agent {agent_name} {'completed' if success else 'failed'} task: {task_name}",
            data={
                "task_name": task_name,
                "success": success
            },
            execution_time_ms=execution_time,
            patient_id=patient_id,
            log_level=LogLevel.INFO if success else LogLevel.ERROR
        )
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary and analytics"""
        session_duration = time.time() - self.session_start_time
        
        return {
            "session_id": self.session_id,
            "session_duration_seconds": session_duration,
            "metrics": self.metrics,
            "events_count": len(self.events),
            "error_rate": self.metrics["error_count"] / max(self.metrics["total_events"], 1),
            "average_execution_time": self._calculate_average_execution_time(),
            "most_common_errors": self._get_most_common_errors(),
            "patient_processing_summary": self._get_patient_processing_summary()
        }
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time for timed events"""
        timed_events = [e for e in self.events if e.execution_time_ms is not None]
        if not timed_events:
            return 0.0
        return sum(e.execution_time_ms for e in timed_events) / len(timed_events)
    
    def _get_most_common_errors(self) -> List[Dict[str, Any]]:
        """Get most common error types"""
        error_events = [e for e in self.events if e.event_type == EventType.ERROR]
        error_components = {}
        
        for event in error_events:
            component = event.component
            if component not in error_components:
                error_components[component] = {"count": 0, "messages": []}
            error_components[component]["count"] += 1
            error_components[component]["messages"].append(event.message)
        
        return [{"component": k, **v} for k, v in 
                sorted(error_components.items(), key=lambda x: x[1]["count"], reverse=True)]
    
    def _get_patient_processing_summary(self) -> Dict[str, Any]:
        """Get summary of patient processing"""
        patient_events = {}
        
        for event in self.events:
            if event.patient_id:
                if event.patient_id not in patient_events:
                    patient_events[event.patient_id] = {
                        "events": 0,
                        "errors": 0,
                        "processing_time": 0,
                        "status": "unknown"
                    }
                
                patient_events[event.patient_id]["events"] += 1
                
                if event.event_type == EventType.ERROR:
                    patient_events[event.patient_id]["errors"] += 1
                
                if event.execution_time_ms:
                    patient_events[event.patient_id]["processing_time"] += event.execution_time_ms
        
        return patient_events
    
    def export_logs(self, filename: Optional[str] = None) -> str:
        """Export all events to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"telemetry_export_{timestamp}_{self.session_id[:8]}.json"
        
        export_data = {
            "session_summary": self.get_session_summary(),
            "events": [asdict(event) for event in self.events]
        }
        
        export_path = self.log_directory / filename
        with open(export_path, 'w') as f:
            json.dump(export_data, f, cls=TelemetryJSONEncoder, indent=2)
        
        return str(export_path)
    
    def cleanup_old_logs(self):
        """Clean up old log files to maintain max_log_files limit"""
        if not self.log_to_file:
            return
        
        log_files = list(self.log_directory.glob("telemetry_*.jsonl"))
        if len(log_files) > self.max_log_files:
            # Sort by creation time and remove oldest
            log_files.sort(key=lambda x: x.stat().st_ctime)
            for old_file in log_files[:-self.max_log_files]:
                old_file.unlink()
    
    def close_session(self):
        """Close the telemetry session"""
        session_summary = self.get_session_summary()
        
        self.log_event(
            event_type=EventType.WORKFLOW_END,
            component="TelemetryCollector",
            message=f"Telemetry session ended: {self.session_id}",
            data=session_summary
        )
        
        # Clean up old logs
        self.cleanup_old_logs()
        
        return session_summary


# Singleton pattern for global telemetry access
class TelemetryManager:
    """Singleton manager for telemetry across the application"""
    _instance = None
    _collector = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelemetryManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, **kwargs):
        """Initialize the telemetry collector"""
        if self._collector is None:
            self._collector = TelemetryCollector(**kwargs)
        return self._collector
    
    def get_collector(self) -> Optional[TelemetryCollector]:
        """Get the current telemetry collector"""
        return self._collector
    
    def shutdown(self):
        """Shutdown telemetry"""
        if self._collector:
            summary = self._collector.close_session()
            self._collector = None
            return summary
        return None


# Convenience functions for easy access
def init_telemetry(**kwargs) -> TelemetryCollector:
    """Initialize telemetry system"""
    manager = TelemetryManager()
    return manager.initialize(**kwargs)


def get_telemetry() -> Optional[TelemetryCollector]:
    """Get current telemetry collector"""
    manager = TelemetryManager()
    return manager.get_collector()


def shutdown_telemetry():
    """Shutdown telemetry system"""
    manager = TelemetryManager()
    return manager.shutdown()