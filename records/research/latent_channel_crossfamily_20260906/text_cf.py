"""Identical sealed V2 text workflow, with endpoint-native fixed-date rendering."""
import sys
import channel_cf as cf

# This process-local binding supplies the new experiment root/endpoints to the
# unchanged sealed text orchestration. No historical file is modified.
sys.modules["channel_v2"] = cf
text = cf.load_pinned("sealed_cf_text", cf.V2 / "text_baseline.py",
                      cf.V2_SEAL["files"]["text_baseline.py"])


def render(tok, body):
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False, date_string="06 Sep 2026")


text.render = render

if __name__ == "__main__":
    text.main()
