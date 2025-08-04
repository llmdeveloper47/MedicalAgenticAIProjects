import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import json
from telemetry import get_telemetry, EventType, LogLevel

# Load environment variables
api_key = ''

client = OpenAI(api_key=api_key)

class Agent:
    """
    A base class for our agents with telemetry integration.
    """
    def __init__(self, name):
        self.name = name
        self.telemetry = get_telemetry()
        
        if self.telemetry:
            self.telemetry.log_event(
                event_type=EventType.AGENT_EXECUTION,
                component=self.name,
                message=f"Agent '{self.name}' initialized",
                data={"agent_type": self.__class__.__name__}
            )
        
        print(f"Agent '{self.name}' initialized.")

    def execute(self, system_prompt: str, user_prompt: str, patient_id: str = None):
        """
        A generic method to execute the agent's task with telemetry.
        Specific agents will override this.
        """
        raise NotImplementedError("Each agent must implement the 'execute' method.")

class ResearchAgent(Agent):
    def __init__(self, name, task_name):
        super().__init__(name)
        self.task_name = task_name
    
    def execute(self, system_prompt: str, user_prompt: str, patient_id: str = None):
        start_time = time.time()
        
        # Log LLM request
        if self.telemetry:
            self.telemetry.log_llm_request(
                agent_name=self.name,
                model="gpt-4",
                prompt_length=len(system_prompt) + len(user_prompt),
                patient_id=patient_id
            )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
            )
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                result_json = json.loads(result_text)
                
                # Log successful LLM response
                if self.telemetry:
                    self.telemetry.log_llm_response(
                        agent_name=self.name,
                        start_time=start_time,
                        response_data=result_json,
                        patient_id=patient_id,
                        success=True
                    )
                    
                    # Log agent execution success
                    self.telemetry.log_agent_execution(
                        agent_name=self.name,
                        task_name=self.task_name,
                        start_time=start_time,
                        success=True,
                        patient_id=patient_id
                    )
                
                return result_json
                
            except json.JSONDecodeError as json_error:
                error_details = {
                    "error_type": "JSONDecodeError",
                    "raw_response": result_text[:500],  # Limit response length in logs
                    "json_error": str(json_error)
                }
                
                print(f"JSON parsing error: {json_error}")
                print(f"Raw response: {result_text}")
                
                # Log JSON parsing error
                if self.telemetry:
                    self.telemetry.log_error(
                        component=self.name,
                        error_message=f"JSON parsing failed: {json_error}",
                        error_details=error_details,
                        patient_id=patient_id
                    )
                    
                    self.telemetry.log_llm_response(
                        agent_name=self.name,
                        start_time=start_time,
                        response_data={"error": str(json_error)},
                        patient_id=patient_id,
                        success=False
                    )
                    
                    self.telemetry.log_agent_execution(
                        agent_name=self.name,
                        task_name=self.task_name,
                        start_time=start_time,
                        success=False,
                        patient_id=patient_id
                    )
                
                return {"error": f"Invalid JSON response: {json_error}"}
                
        except Exception as e:
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "model": "gpt-4",
                "prompt_length": len(system_prompt) + len(user_prompt)
            }
            
            print(f'OpenAI API Error: {e}')
            
            # Log API error
            if self.telemetry:
                self.telemetry.log_error(
                    component=self.name,
                    error_message=f"OpenAI API call failed: {e}",
                    error_details=error_details,
                    patient_id=patient_id
                )
                
                self.telemetry.log_agent_execution(
                    agent_name=self.name,
                    task_name=self.task_name,
                    start_time=start_time,
                    success=False,
                    patient_id=patient_id
                )
            
            return {"error": f"API call failed: {str(e)}"}

class EvaluationAgent(Agent):
    def __init__(self, name, task_name):
        super().__init__(name)
        self.task_name = task_name  

    def execute(self, system_prompt: str, user_prompt: str, patient_id: str = None):
        start_time = time.time()
        
        # Log LLM request
        if self.telemetry:
            self.telemetry.log_llm_request(
                agent_name=self.name,
                model="gpt-4",
                prompt_length=len(system_prompt) + len(user_prompt),
                patient_id=patient_id
            )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
            )
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                result_json = json.loads(result_text)
                
                # Log successful LLM response
                if self.telemetry:
                    self.telemetry.log_llm_response(
                        agent_name=self.name,
                        start_time=start_time,
                        response_data=result_json,
                        patient_id=patient_id,
                        success=True
                    )
                    
                    # Log validation results specifically
                    self.telemetry.log_validation_result(
                        validation_data=result_json,
                        patient_id=patient_id
                    )
                    
                    # Log agent execution success
                    self.telemetry.log_agent_execution(
                        agent_name=self.name,
                        task_name=self.task_name,
                        start_time=start_time,
                        success=True,
                        patient_id=patient_id
                    )
                
                return result_json
                
            except json.JSONDecodeError as json_error:
                error_details = {
                    "error_type": "JSONDecodeError",
                    "raw_response": result_text[:500],  # Limit response length in logs
                    "json_error": str(json_error)
                }
                
                print(f"JSON parsing error: {json_error}")
                print(f"Raw response: {result_text}")
                
                # Log JSON parsing error
                if self.telemetry:
                    self.telemetry.log_error(
                        component=self.name,
                        error_message=f"JSON parsing failed: {json_error}",
                        error_details=error_details,
                        patient_id=patient_id
                    )
                    
                    self.telemetry.log_llm_response(
                        agent_name=self.name,
                        start_time=start_time,
                        response_data={"error": str(json_error)},
                        patient_id=patient_id,
                        success=False
                    )
                    
                    self.telemetry.log_agent_execution(
                        agent_name=self.name,
                        task_name=self.task_name,
                        start_time=start_time,
                        success=False,
                        patient_id=patient_id
                    )
                
                return {"error": f"Invalid JSON response: {json_error}"}
                
        except Exception as e:
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "model": "gpt-4",
                "prompt_length": len(system_prompt) + len(user_prompt)
            }
            
            print(f'OpenAI API Error: {e}')
            
            # Log API error
            if self.telemetry:
                self.telemetry.log_error(
                    component=self.name,
                    error_message=f"OpenAI API call failed: {e}",
                    error_details=error_details,
                    patient_id=patient_id
                )
                
                self.telemetry.log_agent_execution(
                    agent_name=self.name,
                    task_name=self.task_name,
                    start_time=start_time,
                    success=False,
                    patient_id=patient_id
                )
            
            return {"error": f"API call failed: {str(e)}"}
        

class SchedulerAgent:
    """
    Advanced scheduler agent that orchestrates the workflow with intelligent retry logic.
    Handles validation feedback and regenerates invalid categories with targeted prompts.
    """
    
    def __init__(self, name: str = "scheduler_agent", max_retries: int = 5):
        self.name = name
        self.max_retries = max_retries
        self.telemetry = get_telemetry()
        
        # Initialize child agents
        self.research_agent = ResearchAgent("research_agent", "research")
        self.evaluation_agent = EvaluationAgent("evaluation_agent", "evaluation")
        
        # Category mappings for targeted regeneration
        self.categories = {
            "recommendation": "Medical Recommendation",
            "medication_list": "Medication List", 
            "medical_specialtists": "Medical Specialists",
            "considerations": "Considerations"
        }
        
        if self.telemetry:
            self.telemetry.log_event(
                event_type=EventType.AGENT_EXECUTION,
                component=self.name,
                message=f"SchedulerAgent '{self.name}' initialized with max_retries={max_retries}",
                data={
                    "agent_type": self.__class__.__name__,
                    "max_retries": max_retries,
                    "categories": list(self.categories.keys())
                }
            )
    
    def process_patient_workflow(self, patient_data: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """
        Main orchestration method that handles the complete patient workflow
        with intelligent retry and regeneration logic.
        """
        workflow_start_time = time.time()
        
        if self.telemetry:
            self.telemetry.log_event(
                event_type=EventType.WORKFLOW_START,
                component=self.name,
                message=f"Starting intelligent workflow for patient {patient_id}",
                data={
                    "patient_id": patient_id,
                    "workflow_type": "intelligent_retry"
                },
                patient_id=patient_id
            )
        
        try:
            # Step 1: Generate initial recommendation
            print(f"\n🔬 Step 1: Generating Initial Recommendation for {patient_id}")
            initial_recommendation = self._generate_initial_recommendation(patient_data, patient_id)
            
            if "error" in initial_recommendation:
                return self._handle_critical_error("initial_recommendation", initial_recommendation, patient_id)
            
            # Step 2: Validate and regenerate with intelligent retry
            print(f"\n Step 2: Validating and Regenerating (if needed) for {patient_id}")
            final_recommendation = self._validate_and_regenerate(
                patient_data, initial_recommendation, patient_id
            )
            
            # Log successful workflow completion
            workflow_duration = (time.time() - workflow_start_time) * 1000
            if self.telemetry:
                self.telemetry.log_event(
                    event_type=EventType.WORKFLOW_END,
                    component=self.name,
                    message=f"Intelligent workflow completed successfully for patient {patient_id}",
                    data={
                        "patient_id": patient_id,
                        "workflow_successful": True,
                        "total_categories": len(self.categories),
                        "final_recommendation_keys": list(final_recommendation.keys())
                    },
                    execution_time_ms=workflow_duration,
                    patient_id=patient_id,
                    log_level=LogLevel.INFO
                )
            
            return final_recommendation
            
        except Exception as e:
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "patient_id": patient_id
            }
            
            if self.telemetry:
                self.telemetry.log_error(
                    component=self.name,
                    error_message=f"Critical error in workflow for patient {patient_id}: {str(e)}",
                    error_details=error_details,
                    patient_id=patient_id
                )
            
            return {"error": f"Critical workflow error: {str(e)}"}
    
    def _generate_initial_recommendation(self, patient_data: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Generate the initial comprehensive recommendation"""
        
        system_prompt = """You are an expert medical professional in the field of healthcare. Respond only with valid JSON."""
        
        user_prompt = f"""As an expert medical professional in the field of healthcare and specializing in offering expert medical advice based on a patient's medical condition and prior medical history.
        Your task is to understand a patient's prior medical history, current observed medical condition and then suggest a possible treatment plan for the patient to follow as well as the doctor the patient can reach out to.

        Here are the details of the patient:
        
        Patient Information:
        - Age: {patient_data.get('age')}
        - Gender: {patient_data.get('gender')}
        - Medical Condition: {patient_data.get('medical_condition')}
        - Medical History: {patient_data.get('medical_history')}
        
        Instructions:
        1. Create a comprehensive plan to treat the patient based on their current observed medical condition and prior medical history
        2. Adjust the duration of the treatment plan based on the severity of the condition. 
        3. Match treatment plan's solution to the patient's specific medical condition(s)
        4. Suggest treatment options from simpler and easier to follow at home for relief (if applicable) to more complex options offered at a Hospital
        5. Please suggest a list of medication the person can use or discuss with the doctor. This list should be ordered by over the counter medicines to more doctor prescribed medicines
        6. Please include a doctor's specialty type which the patient can reach out to in a clinic for further treatment.

        IMPORTANT: Respond ONLY with a valid JSON object in the following exact format:
        {{
            "recommendation": "the treatment plan the patient needs to follow ranging from simpler options (if applicable) to advanced hospital offered care options",
            "medication_list": "a list of medication the person can use.",
            "medical_specialtists": "A list of possible medical specialist the patient can reach out to for further followups",
            "considerations": "Any additional advice or modifications the client should consider"
        }}

        Do NOT include any text outside the JSON object.
        """
        
        return self.research_agent.execute(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            patient_id=patient_id
        )
    
    def _validate_and_regenerate(self, patient_data: Dict[str, Any], 
                                recommendation: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """
        Core retry logic that validates recommendations and regenerates invalid categories
        """
        current_recommendation = recommendation.copy()
        retry_attempts = {category: 0 for category in self.categories.keys()}
        
        for overall_attempt in range(1, self.max_retries + 1):
            # Validate current recommendation
            validation_result = self._validate_recommendation(
                patient_data, current_recommendation, patient_id, overall_attempt
            )
            
            if "error" in validation_result:
                if self.telemetry:
                    self.telemetry.log_error(
                        component=self.name,
                        error_message=f"Validation failed for patient {patient_id}",
                        error_details=validation_result,
                        patient_id=patient_id
                    )
                return current_recommendation  # Return current state if validation fails
            
            # Check which categories are invalid
            invalid_categories = self._identify_invalid_categories(validation_result)
            
            if not invalid_categories:
                # All categories are valid!
                if self.telemetry:
                    self.telemetry.log_event(
                        event_type=EventType.VALIDATION,
                        component=self.name,
                        message=f"All categories valid for patient {patient_id} after {overall_attempt} overall attempts",
                        data={
                            "patient_id": patient_id,
                            "overall_attempts": overall_attempt,
                            "final_validation": validation_result
                        },
                        patient_id=patient_id
                    )
                print(f" All categories are now valid for patient {patient_id}!")
                return current_recommendation
            
            print(f"🔄 Attempt {overall_attempt}: Found {len(invalid_categories)} invalid categories: {invalid_categories}")
            
            # Regenerate invalid categories
            regeneration_successful = False
            for category in invalid_categories:
                if retry_attempts[category] >= self.max_retries:
                    print(f" Category '{category}' exceeded max retries, marking as 'No Response Found'")
                    current_recommendation[category] = "No Response Found"
                    continue
                
                retry_attempts[category] += 1
                feedback = self._extract_feedback(validation_result, category)
                
                print(f"🔧 Regenerating '{category}' (attempt {retry_attempts[category]}/{self.max_retries})")
                
                regenerated_content = self._regenerate_category(
                    patient_data, category, feedback, current_recommendation, patient_id
                )
                
                if regenerated_content and "error" not in regenerated_content:
                    current_recommendation[category] = regenerated_content
                    regeneration_successful = True
                    
                    if self.telemetry:
                        self.telemetry.log_event(
                            event_type=EventType.LLM_RESPONSE,
                            component=self.name,
                            message=f"Successfully regenerated '{category}' for patient {patient_id}",
                            data={
                                "patient_id": patient_id,
                                "category": category,
                                "attempt": retry_attempts[category],
                                "regenerated_content_length": len(str(regenerated_content))
                            },
                            patient_id=patient_id
                        )
                else:
                    if self.telemetry:
                        self.telemetry.log_error(
                            component=self.name,
                            error_message=f"Failed to regenerate '{category}' for patient {patient_id}",
                            error_details={"category": category, "attempt": retry_attempts[category]},
                            patient_id=patient_id
                        )
            
            if not regeneration_successful:
                print(f" No successful regenerations in attempt {overall_attempt}")
                break
        
        # Mark any remaining invalid categories as "No Response Found"
        final_validation = self._validate_recommendation(patient_data, current_recommendation, patient_id, overall_attempt + 1)
        if "error" not in final_validation:
            final_invalid = self._identify_invalid_categories(final_validation)
            for category in final_invalid:
                print(f" Final marking: '{category}' -> 'No Response Found'")
                current_recommendation[category] = "No Response Found"
        
        return current_recommendation
    
    def _validate_recommendation(self, patient_data: Dict[str, Any], recommendation: Dict[str, Any], 
                               patient_id: str, attempt: int) -> Dict[str, Any]:
        """Validate the current recommendation using the EvaluationAgent"""
        
        system_prompt = """You are an expert medical professional in the field of healthcare, specializing in evaluating and validating expert medical advice based on a patient's medical condition, prior medical history and medical advice as received by the patient from a medical professional. Respond only with valid JSON."""
        
        user_prompt = f"""As an expert medical professional in the field of healthcare and specializing in evaluating and validating expert medical advice based on a patient's medical condition, prior medical history and medical advice as received by the patient from a medical professional.
        Your task is to understand a patient's prior medical history, current observed medical condition, and medical advice as received by the patient from a medical professional and then evaluate if the medical advice received by the patient based on the prior medical history and current observed medical condition is valid or not.
        
        Here are the details of the patient:
        
        Patient Information:
        - Age: {patient_data.get('age')}
        - Gender: {patient_data.get('gender')}
        - Medical Condition: {patient_data.get('medical_condition')}
        - Medical History: {patient_data.get('medical_history')}
        - Medical Recommendation: {recommendation.get('recommendation')}
        - Medication List: {recommendation.get('medication_list')}
        - Medical Specialist: {recommendation.get('medical_specialtists')}
        - Consideration: {recommendation.get('considerations')}
        
        Instructions:
        1. Based on the Age, Medical Condition, Medical History and Initial Medical Recommendation, Validate the Medical Recommendation and answer with a valid vs invalid label. If the response is invalid, Please generate a detailed reasoning of why the original result is invalid given the user's input.
        2. Based on the Age, Medical Condition, Medical History and Initial Medical Recommendation, validate the medication list and answer with a valid vs invalid label. If the list comprises of medications which belong to the same family relevant to the patient's medical condition then it is valid otherwise invalid. If the response is invalid, Please generate a detailed reasoning of why the original result is invalid given the user's input.
        3. Based on the Age, Medical Condition, Medical History and Initial Medical Recommendation, validate the medical specialists list as provided by the previous medical professional and answer with a valid vs invalid label. If any of the medical specialists recommended belong to the current medical condition and prior medical history of the patient, then the label is valid otherwise invalid. If the response is invalid, Please generate a detailed reasoning of why the original result is invalid given the user's input.
        4. Based on the Age, Medical Condition, Medical History and Initial Medical Recommendation, validate the Consideration as provided by the previous medical professional and answer with a valid vs invalid label. If the response is invalid, Please generate a detailed reasoning of why the original result is invalid given the user's input.
        
        IMPORTANT: Respond ONLY with a valid JSON object in the following exact format:
        {{
            "recommendation": "valid or invalid label, if the label is invalid, please also generate a detailed reasoning of why the original result is invalid given the user's input.",
            "medication_list": "valid or invalid label, if the label is invalid, please also generate a detailed reasoning of why the original result is invalid given the user's input.",
            "medical_specialtists": "valid or invalid label, if the label is invalid, please also generate a detailed reasoning of why the original result is invalid given the user's input.",
            "considerations": "valid or invalid label, if the label is invalid, please also generate a detailed reasoning of why the original result is invalid given the user's input."
        }}

        Do NOT include any text outside the JSON object.
        """
        
        if self.telemetry:
            self.telemetry.log_event(
                event_type=EventType.VALIDATION,
                component=self.name,
                message=f"Validating recommendation for patient {patient_id} (attempt {attempt})",
                data={
                    "patient_id": patient_id,
                    "validation_attempt": attempt,
                    "categories_to_validate": list(self.categories.keys())
                },
                patient_id=patient_id
            )
        
        return self.evaluation_agent.execute(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            patient_id=patient_id
        )
    
    def _identify_invalid_categories(self, validation_result: Dict[str, Any]) -> List[str]:
        """Identify which categories are invalid from validation results"""
        invalid_categories = []
        
        for category in self.categories.keys():
            if category in validation_result:
                result = validation_result[category].lower()
                if result.startswith('invalid'):
                    invalid_categories.append(category)
        
        return invalid_categories
    
    def _extract_feedback(self, validation_result: Dict[str, Any], category: str) -> str:
        """Extract detailed feedback for a specific invalid category"""
        if category in validation_result:
            feedback = validation_result[category]
            # Extract the reasoning part after "invalid,"
            if "invalid" in feedback.lower():
                parts = feedback.split(",", 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return "No specific feedback provided"
    
    def _regenerate_category(self, patient_data: Dict[str, Any], category: str, feedback: str,
                           current_recommendation: Dict[str, Any], patient_id: str) -> Optional[str]:
        """
        Regenerate content for a specific invalid category using targeted prompts
        """
        regeneration_start_time = time.time()
        
        if self.telemetry:
            self.telemetry.log_event(
                event_type=EventType.LLM_REQUEST,
                component=self.name,
                message=f"Starting regeneration for category '{category}' for patient {patient_id}",
                data={
                    "patient_id": patient_id,
                    "category": category,
                    "feedback_length": len(feedback),
                    "regeneration_type": "targeted"
                },
                patient_id=patient_id
            )
        
        # Get category-specific prompt
        system_prompt, user_prompt = self._get_category_prompt(category, patient_data, feedback, current_recommendation)
        
        # Use research agent to regenerate
        result = self.research_agent.execute(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            patient_id=patient_id
        )
        
        regeneration_duration = (time.time() - regeneration_start_time) * 1000
        
        if "error" in result:
            if self.telemetry:
                self.telemetry.log_error(
                    component=self.name,
                    error_message=f"Error regenerating category '{category}' for patient {patient_id}",
                    error_details=result,
                    patient_id=patient_id
                )
            return None
        
        # Extract the specific category content
        if isinstance(result, dict) and category in result:
            regenerated_content = result[category]
            
            if self.telemetry:
                self.telemetry.log_event(
                    event_type=EventType.LLM_RESPONSE,
                    component=self.name,
                    message=f"Successfully regenerated category '{category}' for patient {patient_id}",
                    data={
                        "patient_id": patient_id,
                        "category": category,
                        "regenerated_content_length": len(str(regenerated_content)),
                        "used_feedback": feedback[:100]  # First 100 chars of feedback
                    },
                    execution_time_ms=regeneration_duration,
                    patient_id=patient_id
                )
            
            return regenerated_content
        
        return None
    
    def _get_category_prompt(self, category: str, patient_data: Dict[str, Any], 
                           feedback: str, current_recommendation: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate targeted prompts for specific categories based on feedback
        """
        
        base_patient_info = f"""
        Patient Information:
        - Age: {patient_data.get('age')}
        - Gender: {patient_data.get('gender')}
        - Medical Condition: {patient_data.get('medical_condition')}
        - Medical History: {patient_data.get('medical_history')}
        
        Previous Validation Feedback: {feedback}
        """
        
        system_prompt = "You are an expert medical professional. Respond only with valid JSON."
        
        if category == "recommendation":
            user_prompt = f"""As an expert medical professional, generate a comprehensive treatment plan based on the patient information and addressing the feedback provided.
            
            {base_patient_info}
            
            Current recommendation that was marked invalid: {current_recommendation.get('recommendation', 'None')}
            
            Instructions:
            1. Create a comprehensive treatment plan that addresses the validation feedback
            2. Focus on evidence-based medicine appropriate for the patient's age, gender, and condition
            3. Include both conservative and advanced treatment options as appropriate
            4. Ensure the plan is specific to the patient's medical condition and history
            
            IMPORTANT: Respond ONLY with a valid JSON object in this format:
            {{
                "recommendation": "detailed treatment plan that addresses the feedback and is appropriate for this specific patient"
            }}
            """
            
        elif category == "medication_list":
            user_prompt = f"""As an expert medical professional, generate an appropriate medication list based on the patient information and addressing the feedback provided.
            
            {base_patient_info}
            
            Current medication list that was marked invalid: {current_recommendation.get('medication_list', 'None')}
            
            Instructions:
            1. Suggest medications that are specifically relevant to the patient's medical condition
            2. Order from over-the-counter options to prescription medications
            3. Consider the patient's age, gender, and medical history for contraindications
            4. Address the specific concerns mentioned in the validation feedback
            
            IMPORTANT: Respond ONLY with a valid JSON object in this format:
            {{
                "medication_list": "list of appropriate medications ordered from OTC to prescription, relevant to the patient's condition"
            }}
            """
            
        elif category == "medical_specialtists":
            user_prompt = f"""As an expert medical professional, recommend appropriate medical specialists based on the patient information and addressing the feedback provided.
            
            {base_patient_info}
            
            Current specialist recommendations that were marked invalid: {current_recommendation.get('medical_specialtists', 'None')}
            
            Instructions:
            1. Recommend specialists that are directly relevant to the patient's medical condition
            2. Consider the patient's medical history and current symptoms
            3. Ensure specialists can address both the primary condition and related concerns
            4. Address the specific issues mentioned in the validation feedback
            
            IMPORTANT: Respond ONLY with a valid JSON object in this format:
            {{
                "medical_specialtists": "list of appropriate medical specialists directly relevant to the patient's condition and history"
            }}
            """
            
        elif category == "considerations":
            user_prompt = f"""As an expert medical professional, provide important considerations and advice based on the patient information and addressing the feedback provided.
            
            {base_patient_info}
            
            Current considerations that were marked invalid: {current_recommendation.get('considerations', 'None')}
            
            Instructions:
            1. Provide specific advice relevant to the patient's condition and circumstances
            2. Include monitoring requirements, lifestyle modifications, and precautions
            3. Consider the patient's age, gender, and medical history
            4. Address the specific concerns mentioned in the validation feedback
            
            IMPORTANT: Respond ONLY with a valid JSON object in this format:
            {{
                "considerations": "specific advice and considerations relevant to this patient's condition and circumstances"
            }}
            """
        
        else:
            # Fallback for unknown categories
            user_prompt = f"""Generate appropriate content for the category '{category}' based on:
            
            {base_patient_info}
            
            IMPORTANT: Respond ONLY with a valid JSON object in this format:
            {{
                "{category}": "appropriate content for this category"
            }}
            """
        
        return system_prompt, user_prompt
    
    def _handle_critical_error(self, stage: str, error_result: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
        """Handle critical errors that prevent the workflow from continuing"""
        error_message = f"Critical error at stage '{stage}' for patient {patient_id}: {error_result.get('error', 'Unknown error')}"
        
        if self.telemetry:
            self.telemetry.log_error(
                component=self.name,
                error_message=error_message,
                error_details={"stage": stage, "error_result": error_result},
                patient_id=patient_id
            )
        
        print(f" {error_message}")
        return error_result        
