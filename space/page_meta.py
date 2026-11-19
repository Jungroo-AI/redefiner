"""Search and link-preview tags for the demo page.

Gradio's page template hard-codes Gradio's own preview tags (og:title "Gradio", og:url "{url}", Gradio's header image)
and the page has no <title> or description until its JavaScript runs. Link previews (Slack, X, LinkedIn) and crawlers
that do not run JavaScript therefore saw a Gradio page. PageMeta rewrites the HTML of "/" only: it drops every og: and
twitter: tag, adds ours after <meta charset>, and adds a short <noscript> summary with links for readers without
JavaScript.
"""
import html
import json
import re

DROP = re.compile(r'\s*<meta\s+(?:property="og:|name="twitter:|name="description")[^>]*>|\s*<title>.*?</title>', re.S)
CHARSET = re.compile(r"<meta charset=[^>]*>", re.I)
BODY = re.compile(r"<body[^>]*>", re.I)


def e(s):
    return html.escape(s, quote=True)


def head_tags(url, title, description, image, image_alt, links):
    """The tags added to <head>; `links` maps a label to a URL and also feeds the JSON-LD sameAs list."""
    data = {"@context": "https://schema.org", "@type": "WebApplication", "name": "ReDefiNER demo", "url": url,
            "description": description, "applicationCategory": "DeveloperApplication", "operatingSystem": "Web",
            "isAccessibleForFree": True, "image": image,
            "creator": {"@type": "Organization", "name": "Jungroo AI Labs", "url": "https://jungroo.com/"},
            "sameAs": list(links.values())}
    ld = json.dumps(data).replace("</", "<\\/")
    return "\n".join([
        f"<title>{e(title)}</title>",
        f'<meta name="description" content="{e(description)}" />',
        f'<link rel="canonical" href="{e(url)}" />',
        '<meta property="og:type" content="website" />',
        '<meta property="og:site_name" content="ReDefiNER" />',
        f'<meta property="og:url" content="{e(url)}" />',
        f'<meta property="og:title" content="{e(title)}" />',
        f'<meta property="og:description" content="{e(description)}" />',
        f'<meta property="og:image" content="{e(image)}" />',
        '<meta property="og:image:width" content="1200" />',
        '<meta property="og:image:height" content="630" />',
        f'<meta property="og:image:alt" content="{e(image_alt)}" />',
        '<meta name="twitter:card" content="summary_large_image" />',
        f'<meta name="twitter:title" content="{e(title)}" />',
        f'<meta name="twitter:description" content="{e(description)}" />',
        f'<meta name="twitter:image" content="{e(image)}" />',
        f'<meta name="twitter:image:alt" content="{e(image_alt)}" />',
        f'<script type="application/ld+json">{ld}</script>',
    ])


def noscript(title, description, links):
    items = "".join(f'<li><a href="{e(u)}">{e(k)}</a></li>' for k, u in links.items())
    return f"<noscript><h1>{e(title)}</h1><p>{e(description)}</p><ul>{items}</ul></noscript>"


def rewrite(page, head, body):
    """Gradio's page with its preview tags replaced; unchanged if it does not look like Gradio's template."""
    if not CHARSET.search(page):
        return page
    page = CHARSET.sub(lambda m: m.group(0) + "\n" + head, DROP.sub("", page), count=1)
    return BODY.sub(lambda m: m.group(0) + body, page, count=1)


class PageMeta:
    """ASGI middleware: rewrites the HTML that GET "/" returns, and passes every other request through untouched."""

    def __init__(self, app, head, body=""):
        self.app, self.head, self.body = app, head, body

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "GET" or scope["path"] != "/":
            return await self.app(scope, receive, send)
        start, chunks = None, []

        async def capture(message):
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                return
            if message["type"] != "http.response.body" or start is None:
                return await send(message)
            chunks.append(message.get("body", b""))
            if message.get("more_body"):
                return
            headers = [(k, v) for k, v in start["headers"]]
            names = {k.lower(): v for k, v in headers}
            data = b"".join(chunks)
            if (start["status"] == 200 and names.get(b"content-type", b"").startswith(b"text/html")
                    and b"content-encoding" not in names):
                data = rewrite(data.decode("utf-8"), self.head, self.body).encode("utf-8")
                headers = [(k, str(len(data)).encode() if k.lower() == b"content-length" else v) for k, v in headers]
            await send({**start, "headers": headers})
            await send({"type": "http.response.body", "body": data})

        await self.app(scope, receive, capture)
