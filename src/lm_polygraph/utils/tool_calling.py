"""
Tool calling logic for LLM models.
Handles optional and mandatory tool usage.
"""

import logging
import re
import json
from typing import List, Dict, Optional, Tuple

from .tools import ToolManager, Tool, ToolResponse

log = logging.getLogger("lm_polygraph")


def create_tool_selection_prompt(
    question: str,
    tool_manager: ToolManager,
) -> str:
    """
    Create a prompt to ask the LLM whether it wants to use a tool (optional case).
    
    Parameters:
        question (str): Original question.
        tool_manager (ToolManager): Tool manager with available tools.
    
    Returns:
        str: Prompt for tool selection.
    """
    tool_descriptions = tool_manager.get_tool_descriptions()
    
    prompt = f"""You are a helpful assistant. You have access to the following tools:

{tool_descriptions}

Question: {question}

Do you want to use any of these tools to help answer the question? 

If yes, respond in the following JSON format:
{{
    "use_tool": true,
    "tool_name": "<name of the tool>",
    "tool_input": "<input to the tool>"
}}

If no, respond in the following JSON format:
{{
    "use_tool": false
}}

Your response:"""
    
    return prompt


def create_tool_input_prompt(
    question: str,
    tool_manager: ToolManager,
    tool_name: str,
) -> str:
    """
    Create a prompt to ask the LLM for tool input (mandatory case).
    
    Parameters:
        question (str): Original question.
        tool_manager (ToolManager): Tool manager with available tools.
        tool_name (str): Name of the tool that must be used.
    
    Returns:
        str: Prompt for tool input.
    """
    tool = tool_manager.get_tool(tool_name)
    if tool is None:
        raise ValueError(f"Tool {tool_name} not found")
    
    tool_description = tool.get_description()
    
    # For Wikipedia BM25 retriever, we want the question itself as the search query
    if tool_name == "wiki_bm25_retriever" or "wikipedia" in tool_name.lower() or ("wiki" in tool_name.lower() and "bm25" in tool_name.lower()):
        prompt = f"""You are a helpful assistant. You need to use a search tool to find information to answer a question.

Tool: {tool_name}
Description: {tool_description}

Question to answer: {question}

IMPORTANT: Do NOT answer the question. Instead, provide ONLY the search query (keywords or the question itself) that should be used to search for relevant information.

What search query should be used? Provide ONLY the search query, nothing else:

Search query:"""
    else:
        prompt = f"""You are a helpful assistant. You must use the following tool to help answer the question:

Tool: {tool_name}
Description: {tool_description}

Question: {question}

What input should be provided to the tool? Respond with only the tool input (no JSON format, just the input text):

Tool input:"""
    
    return prompt


def create_final_answer_prompt(
    question: str,
    tool_question: str,
    tool_response: str,
    tool_name: str,
) -> str:
    """
    Create a prompt to ask the LLM to answer the original question using tool response.
    
    Parameters:
        question (str): Original question.
        tool_response (str): Response from the tool.
        tool_name (str): Name of the tool that was used.
    
    Returns:
        str: Prompt for final answer.
    """
    import sys
    # Debug: Confirm function is being called
    print(f"[DEBUG create_final_answer_prompt] Called with tool_name={tool_name}, response_length={len(tool_response)}", file=sys.stderr, flush=True)
    
    prompt = f"""You are a helpful assistant. Use the following tool response to answer the question.

Original question: {question}

Question for the tool: {tool_question}
Tool used: {tool_name}
Tool response:
{tool_response}

Original question: {question}
Based on the tool response above, please answer the original question:"""
    
    # Log the enhanced prompt with retrieved documents for debugging
    # Use both logging and stderr to ensure it shows up (Hydra might suppress logs)
    import sys
    
    debug_msg = f"""
{'=' * 80}
ENHANCED PROMPT WITH RETRIEVED DOCUMENTS
{'=' * 80}
Original question: {question}
Tool used: {tool_name}
Tool response (retrieved documents) length: {len(tool_response)} characters
Tool response (retrieved documents) preview (first 500 chars):
{tool_response[:500]}
Full enhanced prompt length: {len(prompt)} characters
Full enhanced prompt:
{'-' * 80}
{prompt}
{'-' * 80}
{'=' * 80}
"""
    
    # Print to stderr (always visible, not suppressed by Hydra)
    print(debug_msg, file=sys.stderr, flush=True)
    
    # Also log it
    log.info("=" * 80)
    log.info("ENHANCED PROMPT WITH RETRIEVED DOCUMENTS")
    log.info("=" * 80)
    log.info(f"Original question: {question}")
    log.info(f"Tool used: {tool_name}")
    log.info(f"Tool response (retrieved documents) length: {len(tool_response)} characters")
    log.info(f"Tool response (retrieved documents) preview (first 500 chars):\n{tool_response[:500]}")
    log.info(f"Full enhanced prompt length: {len(prompt)} characters")
    log.info("Full enhanced prompt:")
    log.info("-" * 80)
    log.info(prompt)
    log.info("-" * 80)
    log.info("=" * 80)
    
    return prompt


def parse_tool_selection_response(response: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Parse the LLM response for tool selection.
    
    Parameters:
        response (str): LLM response.
    
    Returns:
        Tuple[bool, Optional[str], Optional[str]]: (use_tool, tool_name, tool_input)
    """
    # Try to extract JSON from response
    response = response.strip()
    
    # Look for JSON in the response
    json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            use_tool = data.get("use_tool", False)
            if use_tool:
                tool_name = data.get("tool_name")
                tool_input = data.get("tool_input")
                return use_tool, tool_name, tool_input
            else:
                return False, None, None
        except json.JSONDecodeError:
            log.warning(f"Failed to parse JSON from response: {response}")
    
    # Fallback: look for keywords
    if "use_tool" in response.lower() or "tool_name" in response.lower():
        # Try to extract tool name and input from text
        tool_name_match = re.search(r'tool_name["\']?\s*:\s*["\']?([^"\'}\n,]+)', response, re.IGNORECASE)
        tool_input_match = re.search(r'tool_input["\']?\s*:\s*["\']?([^"\'}\n]+)', response, re.IGNORECASE)
        
        if tool_name_match and tool_input_match:
            tool_name = tool_name_match.group(1).strip()
            tool_input = tool_input_match.group(1).strip()
            return True, tool_name, tool_input
    
    # If no tool usage detected, return False
    return False, None, None


def extract_tool_input(response: str, question: str = None, tool_name: str = None) -> str:
    """
    Extract tool input from LLM response (mandatory case).
    
    Parameters:
        response (str): LLM response.
        question (str, optional): Original question (kept for backward compatibility, not used).
        tool_name (str, optional): Name of the tool (kept for backward compatibility, not used).
    
    Returns:
        str: Tool input (cleaned response with prefixes and quotes removed).
    """
    import sys
    try:
        print(f"[DEBUG extract_tool_input] Starting extraction, response: '{response[:100]}'", file=sys.stderr, flush=True)
        # Remove common prefixes
        response = response.strip()
        print(f"[DEBUG extract_tool_input] After strip: '{response[:100]}'", file=sys.stderr, flush=True)
        
        # Remove "Tool input:" prefix if present
        response = re.sub(r'^tool\s*input\s*:?\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        print(f"[DEBUG extract_tool_input] After removing 'tool input:' prefix: '{response[:100]}'", file=sys.stderr, flush=True)
        
        # Remove "Search query:" prefix if present
        response = re.sub(r'^search\s*query\s*:?\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        print(f"[DEBUG extract_tool_input] After removing 'search query:' prefix: '{response[:100]}'", file=sys.stderr, flush=True)
        
        # Remove quotes if present
        if (response.startswith('"') and response.endswith('"')) or \
           (response.startswith("'") and response.endswith("'")):
            response = response[1:-1]
            response = response.strip()
            print(f"[DEBUG extract_tool_input] After removing quotes: '{response[:100]}'", file=sys.stderr, flush=True)
        
        result = response.strip()
        print(f"[DEBUG extract_tool_input] Final result: '{result[:100]}'", file=sys.stderr, flush=True)
        return result
    except Exception as e:
        print(f"[DEBUG extract_tool_input] ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Return the response as-is on error
        return response.strip() if response else ""


def enhance_prompt_with_tool(
    model,
    question: str,
    tool_manager: Optional[ToolManager],
    mandatory: bool = False,
    tool_name: Optional[str] = None,
    max_new_tokens: int = 100,
    **kwargs
) -> Tuple[str, Optional[str], bool]:
    """
    Enhance a prompt by calling a tool and incorporating the tool response.
    This function handles the tool calling workflow but does NOT generate the final answer.
    It only returns the enhanced prompt that includes the tool response.
    
    This is the core tool calling logic shared between:
    - execute_tool_calling_workflow (which then generates the final answer)
    - GreedyProbsCalculator (which uses enhanced prompts for probability calculations)
    
    Parameters:
        model: Model instance (BlackboxModel or WhiteboxModel).
        question (str): Original question.
        tool_manager (Optional[ToolManager]): Tool manager with available tools. If None, returns original question.
        mandatory (bool): Whether tool usage is mandatory.
        tool_name (Optional[str]): Name of the tool to use (required if mandatory).
        max_new_tokens (int): Maximum number of tokens for generation.
        **kwargs: Additional arguments for model generation.
    
    Returns:
        Tuple[str, Optional[str], bool]: (enhanced_prompt, tool_name_used, tool_was_used)
            - enhanced_prompt: Enhanced prompt with tool response (or original question if no tool used)
            - tool_name_used: Name of tool used (None if no tool used)
            - tool_was_used: Whether a tool was actually used
    """
    import sys
    
    print(f"[DEBUG enhance_prompt_with_tool] ===== ENTRY =====", file=sys.stderr, flush=True)
    print(f"[DEBUG enhance_prompt_with_tool] question (first 100 chars): {question[:100]}", file=sys.stderr, flush=True)
    print(f"[DEBUG enhance_prompt_with_tool] tool_manager is None: {tool_manager is None}", file=sys.stderr, flush=True)
    if tool_manager:
        print(f"[DEBUG enhance_prompt_with_tool] tool_manager.has_tools(): {tool_manager.has_tools()}", file=sys.stderr, flush=True)
        if tool_manager.has_tools():
            print(f"[DEBUG enhance_prompt_with_tool] Available tools: {[t.get_name() for t in tool_manager.tools]}", file=sys.stderr, flush=True)
    print(f"[DEBUG enhance_prompt_with_tool] mandatory: {mandatory}, tool_name: {tool_name}", file=sys.stderr, flush=True)
    
    # Backward compatibility: if no tool manager or no tools, return original question
    if not tool_manager or not tool_manager.has_tools():
        print(f"[DEBUG enhance_prompt_with_tool] ===== NO TOOLS AVAILABLE - RETURNING ORIGINAL QUESTION =====", file=sys.stderr, flush=True)
        print(f"[DEBUG enhance_prompt_with_tool] tool_manager is None: {tool_manager is None}", file=sys.stderr, flush=True)
        if tool_manager:
            print(f"[DEBUG enhance_prompt_with_tool] tool_manager.has_tools(): {tool_manager.has_tools()}", file=sys.stderr, flush=True)
        log.debug(f"No tools available: tool_manager={tool_manager}, has_tools={tool_manager.has_tools() if tool_manager else False}")
        return question, None, False
    
    if mandatory:
        print(f"[DEBUG enhance_prompt_with_tool] ===== MANDATORY TOOL MODE =====", file=sys.stderr, flush=True)
        print(f"[DEBUG enhance_prompt_with_tool] tool_name={tool_name}, available tools={[t.get_name() for t in tool_manager.tools]}", file=sys.stderr, flush=True)
        log.debug(f"Mandatory tool mode: tool_name={tool_name}, available tools={[t.get_name() for t in tool_manager.tools]}")
        
        # Mandatory tool usage
        if tool_name is None:
            # If only one tool, use it
            if len(tool_manager.tools) == 1:
                tool_name = tool_manager.tools[0].get_name()
                print(f"[DEBUG enhance_prompt_with_tool] Auto-selected tool_name={tool_name}", file=sys.stderr, flush=True)
            else:
                raise ValueError("tool_name must be specified when mandatory=True and multiple tools are available")
        
        # Step 1: Ask for tool input
        print(f"[DEBUG enhance_prompt_with_tool] Step 1: Creating tool input prompt", file=sys.stderr, flush=True)
        tool_input_prompt = create_tool_input_prompt(question, tool_manager, tool_name)
        print(f"[DEBUG enhance_prompt_with_tool] Step 1: Tool input prompt created (length: {len(tool_input_prompt)} chars)", file=sys.stderr, flush=True)
        
        # Disable tool calling for intermediate steps to avoid recursion
        kwargs_intermediate = kwargs.copy()
        kwargs_intermediate["use_tools"] = False
        print(f"[DEBUG enhance_prompt_with_tool] Step 1: Calling model.generate_texts() with use_tools=False to get tool input", file=sys.stderr, flush=True)
        tool_input_responses = model.generate_texts(
            input_texts=[tool_input_prompt],
            max_new_tokens=max_new_tokens,
            **kwargs_intermediate
        )
        print(f"[DEBUG enhance_prompt_with_tool] Step 1: model.generate_texts() returned, response length: {len(tool_input_responses)}", file=sys.stderr, flush=True)
        if tool_input_responses:
            print(f"[DEBUG enhance_prompt_with_tool] Step 1: Tool input response (first 200 chars): {tool_input_responses[0][:200]}", file=sys.stderr, flush=True)
        
        tool_input = extract_tool_input(tool_input_responses[0], question=question, tool_name=tool_name)
        print(f"[DEBUG enhance_prompt_with_tool] Step 1: Extracted tool_input: {tool_input[:100] if len(tool_input) > 100 else tool_input}", file=sys.stderr, flush=True)
        
        # Step 2: Execute tool
        tool = tool_manager.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        log.info(f"Using tool: {tool_name} (mandatory mode) with input: {tool_input}")
        print(f"Using tool: {tool_name} (mandatory mode) with input: {tool_input}", file=sys.stderr, flush=True)
        tool_response = tool(tool_input)
        
        response_preview = tool_response.content[:200] if len(tool_response.content) > 200 else tool_response.content
        log.info(f"Tool response received (preview): {response_preview}...")
        log.info(f"Tool response full length: {len(tool_response.content)} characters")
        
        # Step 3: Create enhanced prompt with tool response (but don't generate final answer)
        enhanced_prompt = create_final_answer_prompt(question,  tool_input, tool_response.content, tool_name)
        print(f"[DEBUG enhance_prompt_with_tool] Enhanced prompt created (length: {len(enhanced_prompt)} chars)", file=sys.stderr, flush=True)
        return enhanced_prompt, tool_name, True
    
    else:
        # Optional tool usage
        # Step 1: Ask if tool should be used
        selection_prompt = create_tool_selection_prompt(question, tool_manager)
        # Disable tool calling for selection step to avoid recursion
        kwargs_selection = kwargs.copy()
        kwargs_selection["use_tools"] = False
        selection_responses = model.generate_texts(
            input_texts=[selection_prompt],
            max_new_tokens=max_new_tokens,
            **kwargs_selection
        )
        use_tool, selected_tool_name, tool_input = parse_tool_selection_response(selection_responses[0])
        
        if use_tool and selected_tool_name and tool_input:
            # Step 2: Execute tool
            tool = tool_manager.get_tool(selected_tool_name)
            if tool is None:
                log.warning(f"Tool {selected_tool_name} not found, returning original question")
                return question, None, False
            
            log.info(f"Using tool: {selected_tool_name} (optional mode, LLM chose to use tool)")
            tool_response = tool(tool_input)
            
            response_preview = tool_response.content[:200] if len(tool_response.content) > 200 else tool_response.content
            log.info(f"Tool response received (preview): {response_preview}...")
            log.info(f"Tool response full length: {len(tool_response.content)} characters")
            
            # Step 3: Create enhanced prompt with tool response (but don't generate final answer)
            enhanced_prompt = create_final_answer_prompt(question, tool_response.content, selected_tool_name)
            return enhanced_prompt, selected_tool_name, True
        else:
            # No tool usage, return original question
            log.info("No tool used (optional mode, LLM chose not to use tool)")
            return question, None, False


def execute_tool_calling_workflow(
    model,
    question: str,
    tool_manager: Optional[ToolManager],
    mandatory: bool = False,
    tool_name: Optional[str] = None,
    max_new_tokens: int = 100,
    **kwargs
) -> Tuple[str, Optional[str], bool]:
    """
    Execute the tool calling workflow and generate the final answer.
    
    This function uses enhance_prompt_with_tool() to get an enhanced prompt,
    then generates the final answer using that enhanced prompt.
    
    Parameters:
        model: Model instance (BlackboxModel or WhiteboxModel).
        question (str): Original question.
        tool_manager (Optional[ToolManager]): Tool manager with available tools. If None, generates answer directly.
        mandatory (bool): Whether tool usage is mandatory.
        tool_name (Optional[str]): Name of the tool to use (required if mandatory).
        max_new_tokens (int): Maximum number of tokens for generation.
        **kwargs: Additional arguments for model generation.
    
    Returns:
        Tuple[str, Optional[str], bool]: (final_answer, tool_name_used, tool_was_used)
            - final_answer: Final answer from the LLM
            - tool_name_used: Name of tool used (None if no tool used)
            - tool_was_used: Whether a tool was actually used
    """
    import sys
    print(f"[DEBUG] execute_tool_calling_workflow called: tool_manager={tool_manager is not None}, has_tools={tool_manager.has_tools() if tool_manager else False}, mandatory={mandatory}, tool_name={tool_name}", file=sys.stderr, flush=True)
    log.info(f"execute_tool_calling_workflow called: tool_manager={tool_manager}, has_tools={tool_manager.has_tools() if tool_manager else False}, mandatory={mandatory}, tool_name={tool_name}")
    
    # Use the shared enhance_prompt_with_tool function
    enhanced_prompt, tool_name_used, tool_was_used = enhance_prompt_with_tool(
        model=model,
        question=question,
        tool_manager=tool_manager,
        mandatory=mandatory,
        tool_name=tool_name,
        max_new_tokens=max_new_tokens,
        **kwargs
    )
    
    # Generate final answer with the enhanced prompt
    # Disable tool calling for final answer generation to avoid recursion
    kwargs_final = kwargs.copy()
    print(f"[DEBUG] enhanced_prompt: {enhanced_prompt}")
    kwargs_final["use_tools"] = False
    print(f"[DEBUG execute_tool_calling_workflow] Generating final answer with enhanced prompt (length: {len(enhanced_prompt)} chars)", file=sys.stderr, flush=True)
    final_responses = model.generate_texts(
        input_texts=[enhanced_prompt],
        max_new_tokens=max_new_tokens,
        **kwargs_final
    )
    
    return final_responses[0], tool_name_used, tool_was_used

