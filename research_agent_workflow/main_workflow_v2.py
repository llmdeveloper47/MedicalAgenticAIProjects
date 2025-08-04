import sys
import os
sys.path.append('./')
from typing import List, Dict
from workflow_agents.agent_definitions import ResearchAgent, EvaluationAgent, SchedulerAgent
from telemetry import init_telemetry, shutdown_telemetry, EventType, LogLevel


class PatientClass:
    """Represents a patient."""
    def __init__(self, id: str, age: int, gender: str, medical_condition: str, 
                 medical_history: List[str]):
        self.id = id
        self.age = age
        self.gender = gender
        self.medical_condition = medical_condition
        self.medical_history = medical_history
        
    def __str__(self):
        return f"Patient {self.id}: Age {self.age}, Gender: {self.gender}, Condition: {self.medical_condition}"
    
    def to_dict(self) -> Dict:
        """Convert patient data to dictionary for telemetry"""
        return {
            "id": self.id,
            "age": self.age,
            "gender": self.gender,
            "medical_condition": self.medical_condition,
            "medical_history": self.medical_history
        }


def run_intelligent_workflow():
    """
    Run the intelligent workflow with SchedulerAgent that handles
    validation feedback and regeneration automatically.
    """
    # Initialize telemetry system
    telemetry = init_telemetry(
        log_to_file=True,
        log_to_console=True,
        log_directory="logs",
        max_log_files=10
    )
    
    try:
        telemetry.log_event(
            event_type=EventType.WORKFLOW_START,
            component="IntelligentWorkflow",
            message="Starting intelligent patient data workflow with SchedulerAgent",
            data={
                "workflow_version": "2.0",
                "workflow_type": "intelligent_scheduler",
                "total_patients": 2
            }
        )

        # Step 1: Initialize the Patients
        patients = [
            PatientClass(
                id="U001",
                age=35,
                gender='male',
                medical_condition="hypertension with increased sweating and bouts of anxiety and pain in chest",
                medical_history=["increased weight", "history of smoking and alcohol usage", "anxiety medication prescribed"],
            ),
            PatientClass(
                id="U002",
                age=55,
                gender='female',
                medical_condition="insomnia with irritated mood and sudden shifts in user's behaviour. User also shows signs of depression and is allergic to sunlight.",
                medical_history=["history of medication for sleep disorder", "suicidal thoughts found", "recommendation for behavioural therapy"],
            ),
        ]

        # Log patient data input
        for patient in patients:
            telemetry.log_user_input(
                patient_data=patient.to_dict(),
                patient_id=patient.id
            )

        # Step 2: Initialize the SchedulerAgent
        print("\n Initializing Intelligent SchedulerAgent...")
        scheduler_agent = SchedulerAgent(name="intelligent_scheduler", max_retries=5)

        # Step 3: Process each patient through the intelligent workflow
        all_results = []
        
        for patient in patients:
            print(f"\n{'='*80}")
            print(f" Processing {patient}")
            print(f"{'='*80}")
            
            telemetry.log_event(
                event_type=EventType.WORKFLOW_START,
                component="IntelligentWorkflow",
                message=f"Starting intelligent processing for patient {patient.id}",
                data={
                    "patient_id": patient.id,
                    "patient_age": patient.age,
                    "patient_gender": patient.gender,
                    "processing_type": "intelligent_scheduler"
                },
                patient_id=patient.id
            )
            
            # Use SchedulerAgent to handle the complete workflow
            final_recommendation = scheduler_agent.process_patient_workflow(
                patient_data=patient.to_dict(),
                patient_id=patient.id
            )
            
            # Handle potential errors
            if "error" in final_recommendation and final_recommendation["error"]:
                error_message = f"Intelligent workflow failed for patient {patient.id}: {final_recommendation['error']}"
                print(f" {error_message}")
                
                telemetry.log_error(
                    component="IntelligentWorkflow",
                    error_message=error_message,
                    error_details=final_recommendation,
                    patient_id=patient.id
                )
                
                telemetry.log_event(
                    event_type=EventType.WORKFLOW_END,
                    component="IntelligentWorkflow",
                    message=f"Failed to process patient {patient.id}",
                    data={"patient_id": patient.id, "error": final_recommendation['error']},
                    patient_id=patient.id,
                    log_level=LogLevel.ERROR
                )
                
                # Continue with next patient instead of stopping
                continue
            
            # Display results
            print(f"\n Final Results for Patient {patient.id}:")
            print("="*60)
            for category, content in final_recommendation.items():
                status = "SUCCESS" if content != "No Response Found" else "FAILURE"
                print(f"{status} {category.title().replace('_', ' ')}: {content}")
            print("="*60)
            
            # Store results for this patient
            patient_result = {
                "patient_id": patient.id,
                "patient_data": patient.to_dict(),
                "final_recommendation": final_recommendation,
                "processing_method": "intelligent_scheduler"
            }
            all_results.append(patient_result)
            
            # Count successful categories
            successful_categories = sum(1 for content in final_recommendation.values() 
                                      if content != "No Response Found" and "error" not in str(content))
            
            # Log successful patient processing
            telemetry.log_event(
                event_type=EventType.WORKFLOW_END,
                component="IntelligentWorkflow",
                message=f"Completed intelligent processing for patient {patient.id}",
                data={
                    "patient_id": patient.id,
                    "successful_categories": successful_categories,
                    "total_categories": len(final_recommendation),
                    "processing_successful": True,
                    "final_recommendation_summary": {
                        category: "valid" if content != "No Response Found" else "failed"
                        for category, content in final_recommendation.items()
                    }
                },
                patient_id=patient.id,
                log_level=LogLevel.INFO
            )
            
            print(f" Patient {patient.id} processing completed: {successful_categories}/{len(final_recommendation)} categories successful")
        
        # Step 4: Generate overall workflow summary
        total_patients = len(all_results)
        successful_patients = len([r for r in all_results if any(
            content != "No Response Found" for content in r["final_recommendation"].values()
        )])
        
        # Log overall workflow completion
        telemetry.log_event(
            event_type=EventType.WORKFLOW_END,
            component="IntelligentWorkflow",
            message="Intelligent patient data workflow completed successfully",
            data={
                "total_patients_processed": total_patients,
                "successful_patients": successful_patients,
                "workflow_successful": True,
                "processing_method": "intelligent_scheduler",
                "results_summary": {
                    "patient_count": total_patients,
                    "patient_ids": [r["patient_id"] for r in all_results],
                    "success_rate": successful_patients / total_patients if total_patients > 0 else 0
                }
            },
            log_level=LogLevel.INFO
        )
        
        print(f"\n Intelligent Workflow Completed Successfully!")
        print(f" Summary: {successful_patients}/{total_patients} patients processed successfully")
        
        return {
            "status": "Success", 
            "data": all_results, 
            "processed_count": total_patients,
            "successful_count": successful_patients,
            "workflow_type": "intelligent_scheduler"
        }
        
    except Exception as e:
        # Handle any unexpected errors
        error_message = f"Unexpected error in intelligent workflow: {str(e)}"
        print(f" {error_message}")
        
        if telemetry:
            telemetry.log_error(
                component="IntelligentWorkflow",
                error_message=error_message,
                error_details={"exception_type": type(e).__name__, "exception_message": str(e)}
            )
            
            telemetry.log_event(
                event_type=EventType.WORKFLOW_END,
                component="IntelligentWorkflow",
                message="Intelligent workflow failed due to unexpected error",
                data={"error": str(e), "workflow_type": "intelligent_scheduler"},
                log_level=LogLevel.CRITICAL
            )
        
        return {"status": "Critical Error", "details": str(e)}
    
    finally:
        # Always shutdown telemetry and get session summary
        if telemetry:
            session_summary = shutdown_telemetry()
            print(f"\n Telemetry Session Summary")
            print("="*50)
            print(f"Session ID: {session_summary.get('session_id', 'N/A')}")
            print(f"Total Events: {session_summary.get('metrics', {}).get('total_events', 0)}")
            print(f"Error Count: {session_summary.get('metrics', {}).get('error_count', 0)}")
            print(f"LLM Requests: {session_summary.get('metrics', {}).get('llm_requests', 0)}")
            print(f"Validation Attempts: {session_summary.get('metrics', {}).get('validation_attempts', 0)}")
            print(f"Session Duration: {session_summary.get('session_duration_seconds', 0):.2f} seconds")
            
            # Export detailed logs
            export_path = telemetry.export_logs()
            print(f" Detailed logs exported to: {export_path}")


def run_comparison_workflow():
    """
    Run both the original and intelligent workflows for comparison
    """
    print(" Running Comparison Between Original and Intelligent Workflows")
    print("="*80)
    
    # Run intelligent workflow
    print("\n🤖 Running Intelligent Workflow with SchedulerAgent...")
    intelligent_result = run_intelligent_workflow()
    
    print(f"\n Workflow Comparison Results:")
    print("="*50)
    print(f"Intelligent Workflow:")
    print(f"  - Status: {intelligent_result.get('status')}")
    print(f"  - Patients Processed: {intelligent_result.get('processed_count', 0)}")
    print(f"  - Successful Patients: {intelligent_result.get('successful_count', 0)}")
    
    if intelligent_result.get('data'):
        print(f"\n Detailed Results:")
        for patient_result in intelligent_result['data']:
            patient_id = patient_result['patient_id']
            recommendation = patient_result['final_recommendation']
            successful_categories = sum(1 for content in recommendation.values() 
                                      if content != "No Response Found")
            total_categories = len(recommendation)
            
            print(f"  Patient {patient_id}: {successful_categories}/{total_categories} categories successful")
            for category, content in recommendation.items():
                status = "SUCCESS" if content != "No Response Found" else "FAILURE"
                preview = content[:50] + "..." if len(str(content)) > 50 else content
                print(f"    {status} {category}: {preview}")
    
    return intelligent_result


if __name__ == "__main__":
    # Choose which workflow to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--comparison":
        result = run_comparison_workflow()
    else:
        result = run_intelligent_workflow()
    
    print(f"\n Final Result: {result}")