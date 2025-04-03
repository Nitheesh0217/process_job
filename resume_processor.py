import os
import re
import json
import logging
import subprocess
import uuid
from datetime import datetime
from typing import Dict, Any, Set, List, Tuple
from jinja2 import Environment, FileSystemLoader
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

# Constants
ALLOWED_LATEX_COMMANDS = {
    r'\\section', r'\\subsection', r'\\textbf', r'\\textit',
    r'\\begin{itemize}', r'\\end{itemize}', r'\\item',
    r'\\documentclass', r'\\usepackage', r'\\begin{document}',
    r'\\end{document}', r'\\newcommand', r'\\renewcommand',
    r'\\DeclareRobustCommand', r'\\definecolor', r'\\color',
    r'\\pagestyle', r'\\fancyhf', r'\\fancyfoot', r'\\renewcommand',
    r'\\addtolength', r'\\urlstyle', r'\\raggedbottom', r'\\raggedright',
    r'\\setlength', r'\\titleformat', r'\\vspace', r'\\hspace',
    r'\\begin{center}', r'\\end{center}', r'\\begin{tabular*}',
    r'\\end{tabular*}', r'\\href', r'\\texttt', r'\\faPhone',
    r'\\faEnvelope', r'\\faLinkedin'
}

# Dangerous LaTeX patterns
DANGEROUS_PATTERNS = [
    # File system access
    r'\\write18', r'\\input', r'\\include', r'\\openin', r'\\read',
    r'\\write', r'\\openout', r'\\immediate\\write',
    
    # Code execution
    r'\\catcode', r'\\@ifnextchar', r'\\@for', r'\\@whilenum',
    r'\\@forloop', r'\\@whilesw', r'\\@ifundefined',
    
    # Shell escape
    r'\\shellescape', r'\\write18', r'\\immediate\\write18',
    
    # Package loading
    r'\\RequirePackage', r'\\LoadClass', r'\\LoadClassWithOptions',
    
    # System commands
    r'\\system', r'\\exec', r'\\shell', r'\\command',
    
    # File inclusion
    r'\\includegraphics', r'\\includeonly', r'\\includegraphics',
    
    # Dangerous macros
    r'\\def', r'\\edef', r'\\gdef', r'\\xdef', r'\\let',
    r'\\futurelet', r'\\global', r'\\long', r'\\outer',
    
    # Direct LaTeX commands
    r'\\directlua', r'\\luacode', r'\\luaexec',
    
    # Other dangerous commands
    r'\\special', r'\\immediate', r'\\aftergroup', r'\\afterassignment',
    r'\\expandafter', r'\\noexpand', r'\\meaning', r'\\string',
    r'\\csname', r'\\endcsname', r'\\@empty', r'\\@undefined',
    r'\\@tempcnta', r'\\@tempcntb', r'\\@tempdima', r'\\@tempdimb',
    r'\\@tempboxa', r'\\@tempboxb'
]

# Resume JSON Schema
RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "personal_info": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "phone": {"type": "string"},
                "location": {"type": "string"},
                "linkedin": {"type": "string", "format": "uri"}
            },
            "required": ["name", "email"]
        },
        "summary": {"type": "string"},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "position": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "description": {"type": "string"},
                    "achievements": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["company", "position", "start_date", "description"]
            }
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": "string"},
                    "field": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "gpa": {"type": "number"},
                    "honors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["institution", "degree", "field", "start_date"]
            }
        },
        "skills": {
            "type": "object",
            "properties": {
                "technical": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "soft": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["technical"]
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "technologies": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "url": {"type": "string", "format": "uri"}
                },
                "required": ["name", "description"]
            }
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "issuer": {"type": "string"},
                    "date": {"type": "string"},
                    "url": {"type": "string", "format": "uri"}
                },
                "required": ["name", "issuer", "date"]
            }
        }
    },
    "required": ["personal_info", "experience", "education", "skills"]
}

class ResumeProcessor:
    def __init__(self, template_dir: str = "templates"):
        """Initialize the resume processor with template directory."""
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            block_start_string='{%',
            block_end_string='%}',
            variable_start_string='{{',
            variable_end_string='}}',
            autoescape=True
        )

    def _get_user_dir(self, user_id: str) -> str:
        """Get the user's directory path."""
        return os.path.join("users", str(user_id))

    def _get_versions_dir(self, user_id: str) -> str:
        """Get the user's versions directory path."""
        return os.path.join(self._get_user_dir(user_id), "versions")

    def _get_resume_data_path(self, user_id: str) -> str:
        """Get the path to the user's resume data file."""
        return os.path.join(self._get_user_dir(user_id), "resume_data.json")

    def _get_resume_template_path(self, user_id: str) -> str:
        """Get the path to the user's resume template file."""
        return os.path.join(self._get_user_dir(user_id), "resume_template.tex")

    def _get_version_paths(self, user_id: str, timestamp: str) -> Tuple[str, str]:
        """Get the paths for a specific version's tex and pdf files."""
        versions_dir = self._get_versions_dir(user_id)
        return (
            os.path.join(versions_dir, f"{timestamp}.tex"),
            os.path.join(versions_dir, f"{timestamp}.pdf")
        )

    def setup_user_directory(self, user_id: str) -> None:
        """Create the user's directory structure."""
        try:
            # Create main user directory
            user_dir = self._get_user_dir(user_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # Create versions directory
            versions_dir = self._get_versions_dir(user_id)
            os.makedirs(versions_dir, exist_ok=True)
            
            logger.info(f"Created directory structure for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error setting up user directory: {str(e)}")
            raise

    def validate_resume(self, data: dict) -> bool:
        """Validate resume data against the schema."""
        try:
            validate(instance=data, schema=RESUME_SCHEMA)
            return True
        except ValidationError as e:
            logger.error(f"Resume validation error: {str(e)}")
            return False

    def sanitize_resume_data(self, data: dict) -> dict:
        """Sanitize resume data to ensure it meets schema requirements."""
        sanitized = {}
        
        # Personal Info
        if "personal_info" in data:
            sanitized["personal_info"] = {
                "name": str(data["personal_info"].get("name", "")).strip(),
                "email": str(data["personal_info"].get("email", "")).strip(),
                "phone": str(data["personal_info"].get("phone", "")).strip(),
                "location": str(data["personal_info"].get("location", "")).strip(),
                "linkedin": str(data["personal_info"].get("linkedin", "")).strip()
            }
        
        # Summary
        if "summary" in data:
            sanitized["summary"] = str(data["summary"]).strip()
        
        # Experience
        if "experience" in data:
            sanitized["experience"] = []
            for exp in data["experience"]:
                sanitized_exp = {
                    "company": str(exp.get("company", "")).strip(),
                    "position": str(exp.get("position", "")).strip(),
                    "start_date": str(exp.get("start_date", "")).strip(),
                    "end_date": str(exp.get("end_date", "")).strip(),
                    "description": str(exp.get("description", "")).strip(),
                    "achievements": [
                        str(achievement).strip()
                        for achievement in exp.get("achievements", [])
                        if achievement
                    ]
                }
                sanitized["experience"].append(sanitized_exp)
        
        # Education
        if "education" in data:
            sanitized["education"] = []
            for edu in data["education"]:
                sanitized_edu = {
                    "institution": str(edu.get("institution", "")).strip(),
                    "degree": str(edu.get("degree", "")).strip(),
                    "field": str(edu.get("field", "")).strip(),
                    "start_date": str(edu.get("start_date", "")).strip(),
                    "end_date": str(edu.get("end_date", "")).strip(),
                    "gpa": float(edu.get("gpa", 0.0)),
                    "honors": [
                        str(honor).strip()
                        for honor in edu.get("honors", [])
                        if honor
                    ]
                }
                sanitized["education"].append(sanitized_edu)
        
        # Skills
        if "skills" in data:
            sanitized["skills"] = {
                "technical": [
                    str(skill).strip()
                    for skill in data["skills"].get("technical", [])
                    if skill
                ],
                "soft": [
                    str(skill).strip()
                    for skill in data["skills"].get("soft", [])
                    if skill
                ],
                "languages": [
                    str(lang).strip()
                    for lang in data["skills"].get("languages", [])
                    if lang
                ]
            }
        
        # Projects
        if "projects" in data:
            sanitized["projects"] = []
            for proj in data["projects"]:
                sanitized_proj = {
                    "name": str(proj.get("name", "")).strip(),
                    "description": str(proj.get("description", "")).strip(),
                    "technologies": [
                        str(tech).strip()
                        for tech in proj.get("technologies", [])
                        if tech
                    ],
                    "url": str(proj.get("url", "")).strip()
                }
                sanitized["projects"].append(sanitized_proj)
        
        # Certifications
        if "certifications" in data:
            sanitized["certifications"] = []
            for cert in data["certifications"]:
                sanitized_cert = {
                    "name": str(cert.get("name", "")).strip(),
                    "issuer": str(cert.get("issuer", "")).strip(),
                    "date": str(cert.get("date", "")).strip(),
                    "url": str(cert.get("url", "")).strip()
                }
                sanitized["certifications"].append(sanitized_cert)
        
        return sanitized

    async def process_resume_enhancement(self, user_id: str, job_description: str) -> str:
        """Main processing pipeline for resume enhancement"""
        try:
            # Ensure user directory exists
            self.setup_user_directory(user_id)
            
            # Load existing data
            with open(self._get_resume_data_path(user_id), "r") as f:
                resume_data = json.load(f)
            
            # Validate and sanitize resume data
            if not self.validate_resume(resume_data):
                raise ValueError("Invalid resume data structure")
            
            resume_data = self.sanitize_resume_data(resume_data)
            
            # Load template
            with open(self._get_resume_template_path(user_id), "r") as f:
                template = f.read()
            
            # Enhance with AI
            enhanced_data = await self.enhance_with_ai(
                resume_data=resume_data,
                job_description=job_description,
                user_id=user_id
            )
            
            # Validate enhanced data
            if not self.validate_resume(enhanced_data):
                raise ValueError("Invalid enhanced resume data structure")
            
            # Generate LaTeX
            latex_content = self.render_latex(enhanced_data, template)
            
            # Generate timestamp for version
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            
            # Save version files
            tex_path, pdf_path = self._get_version_paths(user_id, timestamp)
            
            # Save LaTeX content
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_content)
            
            # Compile PDF
            self._compile_latex(tex_path)
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error in process_resume_enhancement: {str(e)}")
            raise

    async def enhance_with_ai(self, resume_data: dict, job_description: str, user_id: str) -> dict:
        """Use LLM to enhance resume data with advanced optimization"""
        try:
            prompt = f"""Transform this resume into an ATS-optimized, healthcare AI-focused masterpiece 
            for the following position:
            
            {'-'*40}
            JOB DESCRIPTION:
            {job_description}
            {'-'*40}

            CURRENT RESUME DATA:
            {json.dumps(resume_data, indent=2)}

            Apply these transformation rules with surgical precision:

            1. ATS OPTIMIZATION ENGINE:
            - Extract 15-20 key competencies from job description using TF-IDF analysis
            - Mirror exact terminology from job requirements (e.g., "DICOM integration", "HL7 interfaces")
            - Infuse healthcare AI keywords: clinical NLP, radiology workflow automation, PACS integration
            - Maintain 80%+ keyword density match while keeping natural language flow

            2. CONTEXTUAL ACHIEVEMENT RESTRUCTURING:
            - Convert responsibilities to quantifiable impacts using C-A-R-B format:
              [Challenge] → [Action] → [Result] → [Business Impact]
            - Example transformation:
              "Developed AI models" → "Architected ensemble DL model reducing false positives in chest X-ray analysis by 32%, 
              accelerating radiologist workflow by 19% across 12-hospital network"
            
            3. HEALTHCARE AI DOMAIN SPECIALIZATION:
            - Inject domain-specific terminology:
              • FDA-cleared AI solutions
              • Clinical decision support systems (CDSS)
              • EHR/EMR interoperability
              • Medical imaging preprocessing pipelines
            - Highlight relevant technical stack:
              MONAI, OHIF Viewer, Orthanc DICOM server, FHIR standards

            4. STRUCTURAL ENHANCEMENTS:
            - Prioritize sections: Technical Skills → Healthcare AI Projects → Experience → Education
            - Create dynamic skills matrix mapping to job requirements:
              [Job Skill] → [Years Experience] → [Project Relevance] → [Certifications]
            - Add "Selected Publications/Presentations" section if missing

            5. STYLISTIC OPTIMIZATION:
            - Convert passive voice to power verbs: "Spearheaded", "Architected", "Pioneered"
            - Apply IBM's Resume Likelihood Assessment Framework:
              - 80% concrete nouns → 20% action verbs ratio
              - <5% generic terms ("team player", "detail-oriented")
              - 3-5 bullet points per position with varied length
            
            6. COMPETITIVE DIFFERENTIATION:
            - Add "Healthcare AI Impact Metrics" section:
              • Model deployment scale (# hospitals/clinics)
              • Clinical accuracy improvements
              • Regulatory compliance milestones
            - Include cross-functional leadership achievements
            - Showcase open-source medical AI contributions

            OUTPUT REQUIREMENTS:
            - Maintain strict JSON schema compliance
            - Preserve original LaTeX formatting markers
            - Include ATS readiness score (0-100) in metadata
            - Add "version_notes" with change rationale
            - Flag any potential credibility issues
            
            FINAL CHECK:
            □ All dates in MM/YYYY format
            □ Consistent title capitalization
            □ No first-person pronouns
            □ Company/client confidentiality maintained
            """

            # Use existing API integration
            success, response = send_chat_request(
                prompt=prompt,
                model=user_configs[user_id]['model'],
                api_key=user_api_keys.get(user_id, API_KEY)
            )
            
            if not success:
                raise ValueError("AI enhancement failed")
            
            # Parse and validate enhanced data
            enhanced_data = json.loads(response)
            
            # Validate against schema
            if not self.validate_resume(enhanced_data):
                raise ValueError("Enhanced resume data failed schema validation")
            
            # Sanitize the enhanced data
            enhanced_data = self.sanitize_resume_data(enhanced_data)
            
            # Add metadata
            enhanced_data['metadata'] = {
                'enhancement_timestamp': datetime.now().isoformat(),
                'job_description_hash': hash(job_description),
                'version': '2.0'
            }
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Error in enhance_with_ai: {str(e)}")
            raise ValueError(f"AI enhancement failed: {str(e)}")

    def render_latex(self, data: dict, template: str) -> str:
        """Render JSON data into LaTeX template with custom delimiters"""
        try:
            env = Environment(
                block_start_string='\BLOCK{',
                block_end_string='}',
                variable_start_string='\VAR{',
                variable_end_string='}',
                comment_start_string='\#{',
                comment_end_string='}',
                trim_blocks=True,
                autoescape=False
            )
            
            # Sanitize the data before rendering
            sanitized_data = self._sanitize_data(data)
            
            # Render the template
            rendered = env.from_string(template).render(
                personal=sanitized_data['personal_info'],
                education=sanitized_data['education'],
                skills=sanitized_data['skills'],
                experience=sanitized_data['experience']
            )
            
            # Sanitize the rendered LaTeX
            return self.sanitize_latex(rendered)
            
        except Exception as e:
            logger.error(f"Error rendering LaTeX: {str(e)}")
            raise ValueError(f"Failed to render LaTeX template: {str(e)}")

    def compile_latex(self, content: str, output_dir: str) -> str:
        """Compile LaTeX to PDF with version control and error handling"""
        try:
            # Generate version timestamp
            version = datetime.now().strftime("%Y%m%d-%H%M%S")
            versions_dir = f"{output_dir}/versions"
            os.makedirs(versions_dir, exist_ok=True)
            
            # Define paths
            tex_path = f"{versions_dir}/{version}.tex"
            pdf_path = f"{versions_dir}/{version}.pdf"
            
            # Write LaTeX content
            with open(tex_path, "w", encoding='utf-8') as f:
                f.write(content)
            
            # Compile LaTeX to PDF
            result = subprocess.run([
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory", versions_dir,
                tex_path
            ], capture_output=True, text=True)
            
            # Check for compilation errors
            if result.returncode != 0:
                logger.error(f"LaTeX compilation failed: {result.stderr}")
                raise ValueError(f"LaTeX compilation failed: {result.stderr}")
            
            # Verify PDF was created
            if not os.path.exists(pdf_path):
                raise ValueError("PDF file was not created")
            
            # Clean up auxiliary files
            self._cleanup_aux_files(tex_path)
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error compiling LaTeX: {str(e)}")
            raise ValueError(f"Failed to compile LaTeX: {str(e)}")

    def validate_input(self, text: str) -> bool:
        """Validate user input for safety."""
        # Check for empty or too long input
        if not text or len(text) > 2000:
            return False
            
        # Check for potentially dangerous characters
        dangerous_chars = ['$', '\\', '`', '|', '>', '<', '*']
        if any(char in text for char in dangerous_chars):
            return False
            
        # Check for LaTeX commands
        if re.search(r'\\[a-zA-Z]+\{?', text):
            return False
            
        return True

    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Common technical keywords
        tech_keywords = {
            'languages': ['python', 'java', 'javascript', 'c++', 'typescript'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask'],
            'databases': ['sql', 'mongodb', 'postgresql', 'mysql'],
            'tools': ['git', 'docker', 'kubernetes', 'aws', 'azure'],
            'concepts': ['api', 'rest', 'microservices', 'cloud', 'agile']
        }
        
        found_keywords = []
        text_lower = text.lower()
        
        for category in tech_keywords.values():
            for keyword in category:
                if keyword in text_lower:
                    found_keywords.append(keyword)
                    
        return found_keywords

    def calculate_keyword_match(self, resume_text: str, job_text: str) -> Tuple[float, List[str]]:
        """Calculate keyword match percentage between resume and job description."""
        resume_keywords = set(self.extract_keywords(resume_text))
        job_keywords = set(self.extract_keywords(job_text))
        
        if not job_keywords:
            return 0.0, []
            
        matching_keywords = resume_keywords.intersection(job_keywords)
        missing_keywords = job_keywords - resume_keywords
        
        match_percentage = len(matching_keywords) / len(job_keywords) * 100
        
        return match_percentage, list(missing_keywords)

    def generate_enhancement_suggestions(self, resume_sections: Dict[str, List[str]], job_description: str) -> Dict[str, Any]:
        """Generate suggestions for resume enhancement based on job description."""
        suggestions = {
            "skills_to_add": [],
            "experience_highlights": [],
            "section_improvements": {}
        }
        
        # Extract keywords from job description
        job_keywords = self.extract_keywords(job_description)
        
        # Analyze skills section
        if 'skills' in resume_sections:
            skills_text = ' '.join(resume_sections['skills'])
            _, missing_skills = self.calculate_keyword_match(skills_text, job_description)
            suggestions["skills_to_add"] = missing_skills
        
        # Analyze experience section
        if 'experience' in resume_sections:
            experience_text = ' '.join(resume_sections['experience'])
            for keyword in job_keywords:
                if keyword in experience_text.lower():
                    suggestions["experience_highlights"].append(
                        f"Highlight experience with {keyword}"
                    )
        
        # Generate section-specific improvements
        for section, lines in resume_sections.items():
            section_text = ' '.join(lines)
            match_percentage, _ = self.calculate_keyword_match(section_text, job_description)
            
            if match_percentage < 50:
                suggestions["section_improvements"][section] = [
                    "Consider adding more relevant details",
                    f"Current keyword match: {match_percentage:.1f}%"
                ]
        
        return suggestions

    def sanitize_latex(self, content: str) -> str:
        """Sanitize LaTeX content to prevent malicious commands."""
        try:
            # Remove comments first
            content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
            
            # Remove dangerous patterns
            for pattern in DANGEROUS_PATTERNS:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
            # Split content into lines
            lines = content.split('\n')
            safe_content = []
            
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    safe_content.append(line)
                    continue
                
                # Check for allowed commands
                if any(cmd in line for cmd in ALLOWED_LATEX_COMMANDS):
                    # Verify command structure
                    if self._verify_command_structure(line):
                        safe_content.append(line)
                else:
                    # Remove any LaTeX commands not in allowed list
                    safe_line = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', line)
                    safe_content.append(safe_line)
            
            # Join lines back together
            sanitized_content = '\n'.join(safe_content)
            
            # Final safety checks
            if any(pattern in sanitized_content for pattern in DANGEROUS_PATTERNS):
                raise ValueError("Dangerous LaTeX commands detected after sanitization")
            
            return sanitized_content
            
        except Exception as e:
            logger.error(f"Error in sanitize_latex: {str(e)}")
            raise ValueError("LaTeX sanitization failed")

    def _verify_command_structure(self, line: str) -> bool:
        """Verify that LaTeX commands in the line are properly structured."""
        try:
            # Check for balanced braces
            if line.count('{') != line.count('}'):
                return False
            
            # Check for valid command syntax
            commands = re.finditer(r'\\[a-zA-Z]+(\{.*?\})?', line)
            for cmd in commands:
                cmd_name = re.match(r'\\[a-zA-Z]+', cmd.group(0)).group(0)
                if cmd_name not in ALLOWED_LATEX_COMMANDS:
                    return False
                
                # Check for nested commands
                if '{' in cmd.group(0):
                    nested_cmds = re.findall(r'\\[a-zA-Z]+', cmd.group(0))
                    if any(nested not in ALLOWED_LATEX_COMMANDS for nested in nested_cmds):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in _verify_command_structure: {str(e)}")
            return False

    def extract_latex_sections(self, tex_content: str) -> Dict[str, list]:
        """Parse LaTeX resume into structured sections."""
        sections = {
            'header': [],
            'summary': [],
            'education': [],
            'experience': [],
            'skills': [],
            'projects': [],
            'activities': []
        }
        
        current_section = None
        for line in tex_content.split('\n'):
            if line.strip().startswith('\\section{'):
                section_name = re.search(r'\\section{(.*?)}', line).group(1).lower()
                current_section = section_name if section_name in sections else None
                continue
            
            if current_section and line.strip():
                sections[current_section].append(line)
        
        return sections

    def generate_resume(self, user_id: str, data: Dict[str, Any], template_name: str = "nitheesh_template.tex") -> str:
        """Generate a resume from template and data with enhanced security."""
        try:
            # Create user-specific directory
            user_dir = os.path.join("users", str(user_id))
            os.makedirs(user_dir, exist_ok=True)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tex_file = os.path.join(user_dir, f"resume_{timestamp}.tex")
            pdf_file = os.path.join(user_dir, f"resume_{timestamp}.pdf")

            # Sanitize input data
            sanitized_data = self._sanitize_data(data)

            # Load and render template
            template = self.env.get_template(template_name)
            latex_content = template.render(**sanitized_data)

            # Sanitize generated LaTeX
            safe_content = self.sanitize_latex(latex_content)

            # Write LaTeX file
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(safe_content)

            # Compile LaTeX to PDF
            self._compile_latex(tex_file)

            # Clean up auxiliary files
            self._cleanup_aux_files(tex_file)

            return pdf_file

        except Exception as e:
            logger.error(f"Error generating resume: {str(e)}")
            raise

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data to prevent LaTeX injection."""
        def sanitize_value(value):
            if isinstance(value, str):
                # Escape LaTeX special characters
                value = value.replace('\\', '\\textbackslash{}')
                value = value.replace('{', '\\{')
                value = value.replace('}', '\\}')
                value = value.replace('$', '\\$')
                value = value.replace('&', '\\&')
                value = value.replace('#', '\\#')
                value = value.replace('^', '\\textasciicircum{}')
                value = value.replace('_', '\\_')
                value = value.replace('%', '\\%')
                value = value.replace('~', '\\textasciitilde{}')
                return value
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            return value

        return {k: sanitize_value(v) for k, v in data.items()}

    def _compile_latex(self, tex_file: str) -> None:
        """Compile LaTeX file to PDF with enhanced security."""
        try:
            # Run pdflatex in a secure environment
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-no-shell-escape", tex_file],
                cwd=os.path.dirname(tex_file),
                check=True,
                timeout=30,  # Set timeout to prevent hanging
                capture_output=True,
                text=True
            )
            
            # Check for compilation errors
            if "! LaTeX Error:" in result.stdout or "! Emergency stop." in result.stdout:
                raise RuntimeError(f"LaTeX compilation error: {result.stdout}")
                
            logger.info(f"Successfully compiled {tex_file} to PDF")
            
        except subprocess.TimeoutExpired:
            logger.error(f"LaTeX compilation timed out for {tex_file}")
            raise RuntimeError("LaTeX compilation timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error compiling LaTeX: {str(e)}")
            raise

    def _cleanup_aux_files(self, tex_file: str) -> None:
        """Clean up auxiliary files generated by LaTeX compilation."""
        base_name = os.path.splitext(tex_file)[0]
        aux_extensions = [".aux", ".log", ".out", ".synctex.gz"]
        
        for ext in aux_extensions:
            aux_file = base_name + ext
            if os.path.exists(aux_file):
                try:
                    os.remove(aux_file)
                except Exception as e:
                    logger.warning(f"Error removing auxiliary file {aux_file}: {str(e)}")

    def validate_pdf(self, pdf_path: str) -> bool:
        """Validate generated PDF file."""
        try:
            if not os.path.exists(pdf_path):
                return False
                
            # Check file size (prevent empty or huge files)
            file_size = os.path.getsize(pdf_path)
            if file_size < 100 or file_size > 10_000_000:  # 100B to 10MB
                return False
                
            return True
        except Exception as e:
            logger.error(f"PDF validation error: {str(e)}")
            return False

    def save_resume_version(self, user_id: str, content: str) -> str:
        """Save a version of the resume with metadata."""
        try:
            # Ensure user directory exists
            self.setup_user_directory(user_id)
            
            # Generate timestamp for version
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            tex_path, pdf_path = self._get_version_paths(user_id, timestamp)
            
            # Save LaTeX content
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Compile PDF
            self._compile_latex(tex_path)
            
            # Save metadata
            metadata = {
                "version": timestamp,
                "timestamp": str(datetime.now()),
                "tex_file": tex_path,
                "pdf_file": pdf_path
            }
            
            metadata_path = os.path.join(self._get_versions_dir(user_id), f"{timestamp}.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error saving resume version: {str(e)}")
            raise

    def get_resume_versions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get list of available resume versions for a user."""
        try:
            versions_dir = self._get_versions_dir(user_id)
            if not os.path.exists(versions_dir):
                return []
            
            versions = []
            for filename in os.listdir(versions_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(versions_dir, filename), "r") as f:
                        metadata = json.load(f)
                        versions.append(metadata)
            
            return sorted(versions, key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting resume versions: {str(e)}")
            return []

    def cleanup_old_files(self, user_id: str, max_age_days: int = 30) -> None:
        """Clean up old resume files."""
        try:
            versions_dir = self._get_versions_dir(user_id)
            if not os.path.exists(versions_dir):
                return
            
            current_time = datetime.now()
            for filename in os.listdir(versions_dir):
                file_path = os.path.join(versions_dir, filename)
                file_age = datetime.fromtimestamp(os.path.getctime(file_path))
                
                if (current_time - file_age).days > max_age_days:
                    try:
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old file {file_path}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up old files: {str(e)}") 