from fastchat.conversation import Conversation, SeparatorStyle


def get_vicuna_prompt(input_text: str):
    fs_conv = Conversation(
        name="vicuna_v1.5",
        system_message="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=[["USER", input_text], ["ASSISTANT", ""]],
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )

    return fs_conv.get_prompt()
