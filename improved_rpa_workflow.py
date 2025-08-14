from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from selenium import webdriver
from bs4 import BeautifulSoup
from bs4 import Comment
from bs4.element import Tag, NavigableString
from typing_extensions import TypedDict
from typing import Annotated, Optional, Dict, Any, List
import csv
from langchain_openai import ChatOpenAI  
import os
import json
from datetime import datetime
from pathlib import Path
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import subprocess
import traceback
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_env():
    """从 .env 文件加载环境变量"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key] = value
                        print(f"已加载 {key} = {value[:10]}...")  # 只显示前10个字符以保护隐私
    else:
        print("未找到 .env 文件，请先运行 python setup_env.py 设置 API Keys")
        exit(1)

# 加载环境变量
print("=== 加载环境变量 ===")
load_env()
print(f"DASHSCOPE_API_KEY 状态: {'已设置' if os.environ.get('DASHSCOPE_API_KEY') else '未设置'}")
print(f"TAVILY_API_KEY 状态: {'已设置' if os.environ.get('TAVILY_API_KEY') else '未设置'}")
print(f"LANGSMITH_API_KEY 状态: {'已设置' if os.environ.get('LANGSMITH_API_KEY') else '未设置'}")



class RPAState(TypedDict):
    user_query: Annotated[str, "update_string"]  # 用户原始请求
    target_website: Annotated[Optional[str], "update_string"]  # 解析出的目标网站
    job_keyword: Annotated[Optional[str], "update_string"]  # 解析出的职位关键词
    location: Annotated[Optional[str], "update_string"]  # 解析出的地点
    page_html: Annotated[Optional[str], "update_string"]  # 获取的页面HTML
    filtered_html: Annotated[Optional[str], "update_string"]  # 过滤后的页面HTML
    workflow: dict  # 生成的RPA工作流程
    element_analysis: Annotated[Optional[Dict[str, Any]], "update_string"]  # 元素分析结果
    element_selectors: Annotated[Dict[str, Dict[str, Any]], "update_string"]  # 关键元素定位器
    rpa_script: Annotated[str, "update_string"]  # 生成的RPA脚本
    execution_result: Annotated[Optional[str], "update_string"]  # 脚本执行结果
    error: Annotated[Optional[str], "update_error"]  # 错误信息
    debug_info: Annotated[Optional[str], "update_debug_info"]  # 调试信息

# ===== LLM智能体定义 =====
def site_to_url(site: str) -> str:
    """将常见网站名映射为标准网址"""
    site_map = {
        "51job": "https://www.51job.com",
        "智联招聘": "https://www.zhaopin.com",
        "前程无忧": "https://www.51job.com",
        "猎聘": "https://www.liepin.com",
    }
    if site.startswith("http"):
        return site
    return site_map.get(site, f"https://www.{site}.com")

llm = ChatOpenAI(
    model="qwen-max-0919",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.environ.get("DASHSCOPE_API_KEY")
)

# 需求解析器提示词
query_parser_prompt = ChatPromptTemplate.from_messages([
    ("system", "作为RPA任务解析专家，从用户请求中提取：1)目标网站 2)职位关键词 3)工作地点。输出严格的JSON格式，不要包含其他文本。如果某些信息无法确定，请使用null值。例如：{{target_website: 51job, job_keyword: Java, location: null}}"),
    ("human", "用户请求：{query}")
])

# HTML分析提示词
html_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是有15年前端经验的DOM分析师，请根据网站HTML结构分析以下关键元素：
    {html_content}
    
    目标网站：{target_website}
    任务描述：{task_description}
    
    识别规则：
    1. 搜索输入框：寻找type=text的input元素，特别注意具有placeholder属性的输入框
    2. 地点选择器：寻找包含"城市"、"地点"、"位置"文本的元素，可能是下拉选择框或输入框
    3. 结果容器：包含多个职位信息的区域，通常是一个列表容器
    4. 翻页按钮：包含"下一页"、"next"文本或图标的元素
    5. 搜索按钮：与搜索输入框相关的按钮元素
    
    页面上下文信息：
    请从HTML内容中提取<!-- 语义摘要: -->注释部分的信息，如果没有则填写'无'
    
    请根据上面的HTML内容和上下文信息，分析并返回以下JSON格式。不要使用示例中的默认值，必须根据实际HTML结构进行分析：
    {{
        "search_box": {{"type": "xpath或id或class", "value": "实际的xpath或id或class值", "confidence": 0.0-1.0}},
        "location_input": {{"type": "xpath或id或class", "value": "实际的xpath或id或class值", "confidence": 0.0-1.0}},
        "result_container": {{"type": "xpath或id或class", "value": "实际的xpath或id或class值", "confidence": 0.0-1.0}},
        "pagination": {{"type": "xpath或id或class", "value": "实际的xpath或id或class值", "confidence": 0.0-1.0}},
        "search_button": {{"type": "xpath或id或class", "value": "实际的xpath或id或class值", "confidence": 0.0-1.0}}
    }}
    
    特别注意：
    1. 对于class选择器，请提供完整的class值，不要只提供部分
    2. 对于XPath选择器，请提供完整且准确的XPath表达式
    3. 如果某个元素不存在，请将其值设置为null
    4. confidence表示你对这个选择器准确性的置信度，0.0表示完全不确定，1.0表示完全确定
    5. **重要**：请严格按照JSON格式返回结果，不要包含任何其他文本或解释，不要使用代码块标记（如```json）
    6. **重要**：请确保返回的JSON格式正确，可以被Python的json.loads()函数解析
    7. **重要**：请充分利用提供的上下文信息来提高分析准确性
    """),
    ("human", "请分析目标网站: {target_website}，任务描述: {task_description}")
])

# ===== 节点函数 =====
def parse_user_query(state: RPAState) -> RPAState:
    """解析用户自然语言请求"""
    print("==== 解析用户查询 ====")
    try:
        parser_chain = query_parser_prompt | llm
        response = parser_chain.invoke({"query": state["user_query"]})
        
        # 安全解析JSON
        content = response.content.strip()
        print(f"查询解析LLM原始响应: {content}")
        if content.startswith("```json"):
            content = re.sub(r'^```json\n|\n```$', '', content)
        print(f"查询解析处理后响应: {content}")
        
        parsed = json.loads(content)
        print(f"查询解析结果: {parsed}")
        
        # 修正目标网站
        if "target_website" in parsed:
            parsed["target_website"] = site_to_url(parsed["target_website"])
        
        # 确保必需字段存在
        if "job_keyword" not in parsed or not parsed["job_keyword"]:
            # 尝试从用户查询中提取职位关键词
            user_query = state["user_query"]
            # 常见的职位关键词
            common_keywords = ["Java", "Python", "前端", "后端", "数据分析师", "产品经理", "UI设计师"]
            for keyword in common_keywords:
                if keyword in user_query:
                    parsed["job_keyword"] = keyword
                    break
            
            # 如果还没找到，使用默认值
            if "job_keyword" not in parsed or not parsed["job_keyword"]:
                parsed["job_keyword"] = "Java"  # 默认职位关键词
        
        
        # 显式处理target_website字段，避免重复更新
        new_state = state.copy()
        if "target_website" in parsed:
            new_state["target_website"] = parsed["target_website"]
        if "job_keyword" in parsed:
            new_state["job_keyword"] = parsed["job_keyword"]
        if "location" in parsed:
            new_state["location"] = parsed["location"]
        return new_state
    except Exception as e:
        # 检查是否是API密钥相关的错误
        error_str = str(e)
        if "Arrearage" in error_str or "Access denied" in error_str:
            error_msg = f"API账户欠费或访问被拒绝，请检查您的API密钥和账户状态: {error_str}"
        else:
            error_msg = f"解析请求时出错: {error_str}"
        print(error_msg)
        return {**state, "error": error_msg, "debug_info": traceback.format_exc()}


def fetch_page_html(state: RPAState) -> RPAState:
    """获取目标网站的HTML内容"""
    print("==== 获取页面HTML ====")
    if state.get("error"):
        return state
        
    try:
        target_website = state.get("target_website")
        if not target_website:
            raise ValueError("未指定目标网站")
        
        # 获取网页HTML
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        # 减少资源加载
        prefs = {
            'profile.managed_default_content_settings.images': 2,  # 阻塞图片
            'profile.managed_default_content_settings.font': 2,    # 阻塞字体
            'profile.managed_default_content_settings.media': 2,   # 阻塞媒体文件
            'profile.managed_default_content_settings.stylesheet': 1,  # 允许CSS
            'profile.managed_default_content_settings.javascript': 1   # 允许JS
        }
        options.add_experimental_option('prefs', prefs)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--headless=new")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--enable-unsafe-swiftshader')
        options.add_argument('--use-gl=desktop')
        driver = webdriver.Chrome(service=service, options=options)
        print(f"正在访问网站: {target_website}")
        driver.get(target_website)
        time.sleep(3)  # 确保页面加载
        
        # 获取HTML并关闭驱动
        html = driver.page_source
        driver.quit()
        
        print(f"成功获取页面HTML，长度: {len(html)} 字符")
        return {**state, "page_html": html}
    except Exception as e:
        error_msg = f"获取页面HTML失败: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "debug_info": traceback.format_exc()}

# === 新增：交互元素过滤 ===

def detect_spa(html: str) -> bool:
    """基于启发式判断页面是否为SPA，若是则需要保留script/css以辅助选择器推断。"""
    lowered = html.lower()
    spa_signals = [
        'id="root"', 'id="app"', 'data-reactroot', 'ng-version', 'ng-app', 'v-cloak',
        'next-data', 'vite', 'webpack', 'chunk.js', 'main.js', 'runtime.js', 'app.js',
        'nuxt', 'umi.js', 'single-spa'
    ]
    return any(sig in lowered for sig in spa_signals)


def filter_html_keep_interactive(html: str, preserve_scripts: bool = False, preserve_styles: bool = False) -> str:
    """仅保留与交互相关的DOM节点及必要上下文，可选保留脚本和样式，清理无关标签与冗余属性。"""
    soup = BeautifulSoup(html, "html.parser")

    remove_tags = [
        # 根据参数决定是否移除脚本/样式/样式链接
        *([] if preserve_scripts else ["script"]),
        *([] if preserve_styles else ["style", "link"]),
        # 始终移除的
        "svg", "img", "picture", "source", "video", "audio",
        "canvas", "iframe", "object", "embed", "noscript", "meta", "path"
    ]
    if remove_tags:
        for tag in soup.find_all(remove_tags):
            tag.decompose()

    summary_comments_texts: List[str] = []
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "语义摘要" in str(comment):
            summary_comments_texts.append(str(comment))
        comment.extract()

    interactive_roles = {
        "button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio",
        "switch", "slider", "menuitem", "menu", "tab", "tabpanel", "tablist",
        "listbox", "option", "spinbutton", "treeitem"
    }

    def is_interactive(tag) -> bool:
        if not isinstance(tag, Tag):
            return False
        name = tag.name.lower()
        if name in {"input", "button", "select", "textarea", "option", "optgroup", "datalist", "form", "label"}:
            return True
        if name == "a" and (tag.has_attr("href") or (tag.get("role") in interactive_roles)):
            return True
        role = tag.get("role")
        if role and role.lower() in interactive_roles:
            return True
        tabindex = tag.get("tabindex")
        if tabindex is not None and str(tabindex).strip() != "":
            try:
                return int(str(tabindex)) >= 0
            except Exception:
                return True
        if tag.has_attr("contenteditable"):
            value = str(tag.get("contenteditable")).lower()
            if value != "false":
                return True
        for attr_name in tag.attrs.keys():
            if isinstance(attr_name, str) and attr_name.startswith("on") and len(attr_name) > 2:
                return True
        aria_attrs = [k for k in tag.attrs.keys() if isinstance(k, str) and k.startswith("aria-")]
        if aria_attrs:
            return True
        return False

    tags_to_keep = set()
    interactive_tags = []
    for t in soup.find_all(True):
        if is_interactive(t):
            interactive_tags.append(t)
            tags_to_keep.add(t)
            for anc in t.parents:
                if isinstance(anc, Tag):
                    tags_to_keep.add(anc)
                    if anc.name in ("body", "html"):
                        break

    def keep_label_for_control(control):
        control_id = control.get("id")
        if not control_id:
            return
        label = soup.find("label", attrs={"for": control_id})
        if label:
            tags_to_keep.add(label)
            for anc in label.parents:
                if isinstance(anc, Tag):
                    tags_to_keep.add(anc)
                    if anc.name in ("body", "html"):
                        break

    for t in interactive_tags:
        if t.name in {"input", "select", "textarea"}:
            keep_label_for_control(t)

    neighbor_text_tags = {"span", "small", "strong", "em", "b", "i", "u", "p", "label", "h1", "h2", "h3", "h4", "h5", "h6"}
    for t in interactive_tags:
        for sib in [t.previous_sibling, t.next_sibling]:
            if sib is None:
                continue
            if isinstance(sib, Tag) and sib.name in neighbor_text_tags:
                tags_to_keep.add(sib)
                for anc in sib.parents:
                    if isinstance(anc, Tag):
                        tags_to_keep.add(anc)
                        if anc.name in ("body", "html"):
                            break

    for t in soup.find_all(True):
        if t.name in ("html", "head", "body"):
            continue
        if t not in tags_to_keep:
            t.decompose()

    # 允许的属性集合，扩展以支持脚本/样式/样式链接的关键属性
    allowed_attrs = {
        "id", "class", "name", "role", "href", "type", "placeholder", "value", "for",
        "title", "alt", "aria-label", "aria-labelledby", "aria-describedby",
        "autocomplete", "tabindex", "style",
        # script/link/style supportive attributes
        "src", "rel", "media", "async", "defer", "crossorigin", "integrity", "nomodule"
    }
    allowed_event_attrs = {"onclick", "onchange", "oninput", "onkeydown", "onkeyup", "onsubmit"}
    allowed_data_attrs = {"data-test", "data-testid", "data-qa", "data-automation", "data-qaid", "data-cy", "data-id"}

    for t in soup.find_all(True):
        if not isinstance(t, Tag):
            continue
        attrs = dict(t.attrs)
        for attr_name in list(attrs.keys()):
            if attr_name in allowed_attrs or attr_name in allowed_event_attrs or attr_name in allowed_data_attrs:
                continue
            if attr_name.startswith("aria-"):
                continue
            del t.attrs[attr_name]

    # 压缩文本节点
    for text_node in soup.find_all(string=True):
        if isinstance(text_node, Comment):
            continue
        # 跳过script/style内文本（后续单独处理）
        if text_node.parent and text_node.parent.name in ("script", "style"):
            continue
        raw = str(text_node)
        stripped = raw.strip()
        if not stripped:
            text_node.replace_with("")
            continue
        normalized = re.sub(r"\s+", " ", stripped)
        if len(normalized) > 200:
            normalized = normalized[:200] + "…"
        if normalized != raw:
            text_node.replace_with(NavigableString(normalized))

    # 对保留的内联script/style进行内容截断，避免巨大体积
    if preserve_scripts:
        for s in soup.find_all("script"):
            # 仅截断内联脚本内容
            if not s.has_attr("src") and s.string:
                content = str(s.string)
                if len(content) > 8000:
                    s.string.replace_with(content[:8000] + "/* …truncated… */")
    if preserve_styles:
        for st in soup.find_all("style"):
            if st.string:
                content = str(st.string)
                if len(content) > 20000:
                    st.string.replace_with(content[:20000] + "/* …truncated… */")

    if summary_comments_texts:
        body = soup.body or soup
        for text in reversed(summary_comments_texts):
            body.insert(0, Comment(text))

    html_out = str(soup)
    html_out = re.sub(r">\s+<", "><", html_out)
    html_out = re.sub(r"\s{2,}", " ", html_out)
    return html_out


def filter_interactive_html(state: RPAState) -> RPAState:
    """工作流节点：过滤HTML，仅保留交互相关元素；对SPA或指示需要CSS的页面保留脚本/样式。"""
    print("==== 过滤HTML只保留交互相关元素 ====")
    if state.get("error"):
        return state
    try:
        html = state.get("page_html", "")
        if not html:
            return state
        # 环境变量覆盖（若用户强制指定）
        env_keep_scripts = os.environ.get("FILTER_KEEP_SCRIPTS", "").lower() in ("1", "true", "yes")
        env_keep_styles = os.environ.get("FILTER_KEEP_STYLES", "").lower() in ("1", "true", "yes")
        spa = detect_spa(html)
        preserve_scripts = env_keep_scripts or spa
        preserve_styles = env_keep_styles or spa
        filtered = filter_html_keep_interactive(html, preserve_scripts=preserve_scripts, preserve_styles=preserve_styles)
        print(f"过滤后HTML长度: {len(filtered)}，压缩率: {len(filtered) / max(len(html), 1):.2%} (SPA={spa}, keep_scripts={preserve_scripts}, keep_styles={preserve_styles})")
        return {**state, "filtered_html": filtered}
    except Exception as e:
        error_msg = f"HTML过滤失败: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "debug_info": traceback.format_exc()}


def analyze_html_chunk(chunk: str, target_website: str, user_query: str) -> dict:
    """基于HTML分片进行元素分析"""
    try:
        analyzer_chain = html_analyzer_prompt | llm
        response = analyzer_chain.invoke({
            "html_content": chunk[:2000],  # 截断避免token溢出
            "target_website": target_website,
            "task_description": user_query
        })
        
        # 安全解析JSON
        content = response.content.strip()
        if content.startswith("```json"):
            content = re.sub(r'^```json\n|\n```$', '', content)
        
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}\n原始响应内容:\n{content}")
        return {"error": "元素分析结果解析失败"}
    except Exception as e:
        print(f"元素分析异常: {str(e)}")
        return {"error": "元素分析过程出错"}

def analyze_page_elements(state: RPAState) -> RPAState:
    """基于滑动窗口的HTML分片元素分析"""
    print("==== 执行元素分析 ====")
    if state.get("error"):
        return state

    try:
        html_content = state.get("filtered_html") or state["page_html"]
        window_size = 3000  # 字符窗口大小
        overlap_size = 500   # 重叠区域
        chunks = [html_content[i:i+window_size] 
                 for i in range(0, len(html_content), window_size - overlap_size)]

        element_cache = {
            'search_box': {'candidates': [], 'best_score': 0.0},
            'location_input': {'candidates': [], 'best_score': 0.0},
            'result_container': {'candidates': [], 'best_score': 0.0}
        }

        for idx, chunk in enumerate(chunks):
            print(f"分析分片 {idx+1}/{len(chunks)} (长度: {len(chunk)})")
            
            # 调用LLM分析分片
            analysis = analyze_html_chunk(
                chunk,
                state["target_website"],
                state["user_query"]
            )

            # 融合分片结果
            for element_type in element_cache:
                if analysis.get(element_type):
                    element_cache[element_type]['candidates'].append({
                        'selector': analysis[element_type]['value'],
                        'confidence': analysis[element_type]['confidence'],
                        'chunk_index': idx
                    })
                    
                    # 更新最佳选择
                    if analysis[element_type]['confidence'] > element_cache[element_type]['best_score']:
                        element_cache[element_type]['best_score'] = analysis[element_type]['confidence']
                        element_cache[element_type]['best_selector'] = analysis[element_type]['value']

        # 构建最终结果
        final_selectors = {
            element_type: {
                'selector': cache['best_selector'],
                'confidence': cache['best_score'],
                'source_chunks': [c['chunk_index'] for c in cache['candidates']
                                if c['confidence'] > 0.7]
            }
            for element_type, cache in element_cache.items()
        }

        print("元素定位器结果:")
        for elem, data in final_selectors.items():
            print(f"{elem}: {data['selector']} (置信度: {data['confidence']:.2f})")

        return {**state, "element_selectors": final_selectors}

    except Exception as e:
        error_msg = f"元素分析失败: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "debug_info": traceback.format_exc()}

def generate_workflow(state: RPAState) -> RPAState:
    """生成包含子流程的RPA工作流程"""
    print("==== 生成RPA流程 ====")
    if state.get("error"):
        return state

    try:
        user_task = state["user_query"]
        elements_json = json.dumps(state["element_selectors"], indent=2)
        target_website=state["target_website"]
        # 构造LLM提示词
        workflow_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            您是一个专业的RPA流程设计师，正在处理{target_website}的自动化任务。

            用户任务:
            {user_task}

            网站交互元素信息:
            {elements_json}

            请设计RPA操作流程步骤:
            1. 按步骤描述操作序列（1个主流程 + 3个子流程）
            2. 每个步骤包含: 动作、目标元素、参数、描述
            3. 添加异常处理和重试机制
            4. 包含数据收集、页面导航和结果保存
            5. 输出严格的JSON格式
            """),
            ("human", "请生成符合要求的JSON工作流程")
        ])

        # 调用大模型生成流程
        chain = workflow_prompt | llm
        response = chain.invoke({
            "user_task": user_task,
            "elements_json": elements_json,
            "target_website":target_website
        })

        # 解析生成的流程
        workflow = json.loads(response.content)
        
        # 构建流程字典
        workflow_dict = {
            "main_flow": {
                "steps": workflow["main_steps"],
                "retry_policy": {
                    "max_attempts": 3,
                    "delay_seconds": 5
                }
            },
            "sub_flows": {
                "data_collection": workflow["data_collection_steps"],
                "pagination": workflow["pagination_steps"],
                "error_handling": workflow["error_handling_steps"]
            },
            "output_spec": {
                "csv_columns": ["职位名称", "公司名称", "薪资范围", "发布时间"],
                "save_path": "./job_results.csv"
            }
        }

        return {**state, "workflow": workflow_dict}

    except json.JSONDecodeError as e:
        error_msg = f"工作流程解析失败: {str(e)}"
        print(f"原始响应内容:\n{response.content}")
        return {**state, "error": error_msg}
    except Exception as e:
        error_msg = f"流程生成失败: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "debug_info": traceback.format_exc()}
       
        
        
        


def generate_rpa_script(state: RPAState) -> RPAState:
    """使用大模型生成RPA脚本"""
    print("==== 生成RPA脚本 ====")
    if state.get("error"):
        return state

    try:
        # 构造代码生成提示词
        code_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            你是有10年经验的RPA开发专家，请根据以下信息生成包含main函数的完整Selenium脚本：
            用户请求：{user_query}
            目标网站：{target_website}
            职位关键词：{job_keyword}
            工作地点：{location}
            元素定位信息：{element_selectors}
            RPA流程信息:{workflow}
            
            要求：
            1. 必须包含def main()入口函数
            2. main函数返回职位信息列表
            3. 使用Python的Selenium库
            4. 包含浏览器初始化配置
            5. 使用提供的元素定位器
            6. 实现完整搜索流程
            7. 添加异常处理机制
            8. 结果保存到CSV
            9. 必须包含详细的数据提取日志
            10. 每个数据提取步骤添加重试机制
            11. 添加显式的等待条件
            
            示例结构：
            ```python
            from selenium import webdriver
            from selenium.webdriver.support.ui import WebDriverWait
            
            def main():
                # 在ChromeOptions配置后添加驱动版本指定
                service = Service(ChromeDriverManager().install())
                options = webdriver.ChromeOptions()
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--headless=new")
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--ignore-certificate-errors")
                options.add_argument('--allow-running-insecure-content')
                options.add_argument('--enable-unsafe-swiftshader')
                options.add_argument('--use-gl=desktop')
                driver = webdriver.Chrome(service=service, options=options)
                
                driver.get(target_website)
                time.sleep(3) 
                        
                try:
                    # 带显式等待的元素定位
                    elements = WebDriverWait(driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".job-list .item"))
                    )
                    print(f"找到{{len(elements)}}条职位数据")
                    
                    # 带重试的数据提取
                    for i in range(3):  # 最多重试3次
                        try:
                            # 数据提取逻辑
                            break
                        except Exception as e:
                            print(f"第{{i+1}}次重试失败: {{str(e)}}")
                    
                    return results
                finally:
                    driver.quit()
            
            if __name__ == '__main__':
                main()
            ```
            """),
            ("human", "请生成符合上述要求的完整脚本")
        ])

        # 获取必要参数
        params = {
            "user_query": state["user_query"],
            "target_website": state.get("target_website", ""),
            "job_keyword": state.get("job_keyword", ""),
            "location": state.get("location", ""),
            "element_selectors": json.dumps(state.get("element_selectors", {}), indent=2)
        }

        # 调用大模型生成代码
        code_chain = code_prompt | llm
        response = code_chain.invoke(params)
        generated_code = response.content

        # 提取代码块
        code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
        if not code_match:
            raise ValueError("未找到有效的Python代码块")

        # 保存生成的脚本
        final_code = code_match.group(1)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_rpa_script_{timestamp}.py"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_code)
        print(f"RPA脚本已保存至：{os.path.abspath(filename)}")
        return {**state, "rpa_script": final_code}

    except Exception as e:
        error_msg = f"脚本生成失败: {str(e)}"
        print(error_msg)
        return {**state, "error": error_msg, "debug_info": traceback.format_exc()}
        
    

def execute_rpa_script(state: RPAState) -> RPAState:
    """动态执行生成的RPA脚本"""
    print("==== 执行RPA脚本 ====")
    if state.get("error"):
        return state

    try:
        # 创建临时文件保存脚本
        script_content = (
            "from selenium import webdriver\n"
            "from selenium.webdriver.common.by import By\n"
            "from selenium.webdriver.support.ui import WebDriverWait\n"
            "from selenium.webdriver.support import expected_conditions as EC\n"
            "import time\n"
            "import csv\n"
            "import logging\n\n"
            "logging.basicConfig(level=logging.INFO)\n\n"
            f"{state['rpa_script']}"
        )
        
        
        # 执行动态脚本
        exec_globals = {}
        exec(script_content, exec_globals)
        
        # 获取执行结果
        if 'main' in exec_globals:
            print("正在执行脚本的main函数...")
            results = exec_globals['main']()
            if not results:
                raise ValueError("脚本返回空数据集")
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"job_results_{timestamp}.csv"
            
            # 保存结果到CSV
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['职位名称', '公司名称', '工作地点', '薪资范围', '发布时间'])
                
                if isinstance(results, list) and len(results) > 0:
                    writer.writerows(results)
                    print(f"成功保存{len(results)}条职位信息到{filename}")
                else:
                    print("警告: 未保存任何数据")
                    open(filename, 'w').close()  # 创建空文件
            
            return {**state, "execution_result": filename}
        
        raise ValueError("生成的脚本中缺少main函数")
    
    except Exception as e:
        error_msg = f"脚本执行失败: {str(e)}"
        print(error_msg)
        print("==== 详细错误信息 ====")
        print(traceback.format_exc())
        return {**state, "error": error_msg, "debug_info": traceback.format_exc()}
        
    
# ===== 工作流编排 =====
def has_error(state: RPAState) -> bool:
    return state.get("error") is not None

def error_handler(state: RPAState) -> RPAState:
    return {**state, "error": f"流程中断: {state.get('error')}"}

# 创建工作流
builder = StateGraph(RPAState)
builder.add_node("query_parser", parse_user_query)
builder.add_node("page_fetcher", fetch_page_html)
builder.add_node("html_filter", filter_interactive_html)
builder.add_node("element_analyzer", analyze_page_elements)
builder.add_node("script_generator", generate_rpa_script)
builder.add_node("script_executor", execute_rpa_script)
builder.add_node("error_handler", error_handler)

builder.set_entry_point("query_parser")

# 添加边
builder.add_edge("query_parser", "page_fetcher")
builder.add_edge("page_fetcher", "html_filter")
builder.add_edge("html_filter", "element_analyzer")
builder.add_edge("element_analyzer", "script_generator")
builder.add_edge("script_generator", "script_executor")

# 添加条件边
builder.add_conditional_edges(
    "query_parser",
    lambda state: "error_handler" if has_error(state) else "page_fetcher"
)
builder.add_conditional_edges(
    "page_fetcher",
    lambda state: "error_handler" if has_error(state) else "html_filter"
)
builder.add_conditional_edges(
    "html_filter",
    lambda state: "error_handler" if has_error(state) else "element_analyzer"
)
builder.add_conditional_edges(
    "element_analyzer",
    lambda state: "error_handler" if has_error(state) else "script_generator"
)
builder.add_conditional_edges(
    "script_generator",
    lambda state: "error_handler" if has_error(state) else "script_executor"
)
builder.add_conditional_edges(
    "script_executor",
    lambda state: "error_handler" if has_error(state) else END
)

# 错误处理边
builder.add_edge("error_handler", END)

improved_rpa_agent = builder.compile(checkpointer=MemorySaver())

# ===== 主执行程序 =====
if __name__ == "__main__":
    user_query = "从51job网站搜索北京的java开发职位"
    print(f"用户查询: {user_query}")
    state = {"user_query": user_query}
    
    print("==== 开始执行改进的工作流 ====")
    final_state = None
    
    try:
        # 执行工作流
        for step in improved_rpa_agent.stream(
            state,
            config={"configurable": {"thread_id": "improved-test-thread-001"}}
        ):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            
            print(f"==== 节点: {node_name} ====")
            if node_state.get("error"):
                print(f"错误: {node_state['error']}")
                if node_state.get("debug_info"):
                    print("=== 调试信息 ===")
                    print(node_state["debug_info"])
            
            # 保存最终状态
            final_state = node_state
    
    except Exception as e:
        print(f"工作流执行失败: {str(e)}")
        traceback.print_exc()
    
    print("==== 工作流执行结束 ====")
    if final_state and final_state.get("error"):
        print(f"最终状态: 失败 - {final_state['error']}")
    elif final_state and "execution_result" in final_state:
        print("最终状态: 成功执行脚本")
        print("执行结果:")
        print(final_state["execution_result"])
    else:
        print("最终状态: 未知")



