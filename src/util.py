import json
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

def inference_by_vllm(
    model_path: str,
    dataset_path: str,
    output_path: str,
    lora_path: str = None,
):
    """Use vLLM for efficient batched inference with optional LoRA adapters. Write results to output file."""
    dataset = json.load(open(dataset_path, "r", encoding="utf-8"))
    prompts = [f"{item['instruction']}\n问题：{item['input']}\n回答：" for item in dataset]
    
    # generate responses
    write_data = {}

    engine_args = EngineArgs(
        model=model_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=8,
        max_cpu_loras=2,
        max_num_seqs=256,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )
    lora_request = None
    if lora_path is not None:
        lora_request = LoRARequest("lora", 1, lora_path)
    
    request_id = 0
    while request_id < len(prompts) or engine.has_unfinished_requests():
        if request_id < len(prompts):
            prompt = prompts[request_id]
            engine.add_request(
                str(request_id), prompt, sampling_params, lora_request=lora_request
            )
            write_data[str(request_id)] = {"input": prompt}
            request_id += 1
        
        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                if int(request_output.request_id) % 100 == 0:
                    print(f"req id {request_output.request_id} done")
                
                # 获取生成的文本并处理空答案
                response = request_output.outputs[0].text.strip()
                if not response:
                    response = "无法生成回答"
                
                write_data[request_output.request_id]["output"] = response
            

    # write to output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(write_data, f, ensure_ascii=False, indent=4)

    return write_data