import os
import shutil
import yaml
from pathlib import Path


from ba_ragmas_chatbot.graph.workflow import create_graph
from ba_ragmas_chatbot.tools.vectorstore import setup_vectorstore

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackContext,
    ConversationHandler,
    CallbackQueryHandler,
    filters,
)

from langchain_ollama import OllamaLLM
from ba_ragmas_chatbot import logger_config
from ba_ragmas_chatbot.states import S
from ba_ragmas_chatbot.paths import DOCUMENTS_DIR
from ba_ragmas_chatbot.paths import DB_DIR


class TelegramBot:

    CHAT = -1

    VALID_MIME_TYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]

    def __init__(self):
        self.logger = logger_config.get_logger("telegram bot")
        self.config = self._load_config()
        self.token = os.getenv("TELEGRAM_TOKEN")

        if not self.token:
            self.token = self.config.get("chatbot_token", {}).get("token")
        if not self.token:
            raise ValueError(
                "❌ no telegram token found! please set in .env or configs.yaml."
            )
        model_cfg = self.config.get("models", {})
        self.llm_name = model_cfg.get("chat_model", "llama3.1:8b-instruct-q8_0")
        self.llm_url = model_cfg.get("base_url", "http://localhost:11434")
        self.ai = OllamaLLM(model=self.llm_name, base_url=self.llm_url)
        self.tools = []

    def _load_config(self):
        """Lädt die Konfiguration robust relativ zum Modulpfad."""
        here = Path(__file__).resolve().parent
        config_path = here / "config" / "configs.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def clear_db(self):
        """Deletes all database files related to ChromaDB."""
        db_folder = str(DB_DIR)
        if os.path.exists(db_folder):
            shutil.rmtree(db_folder)
        os.makedirs(db_folder, exist_ok=True)

    # navigation + keyboards

    def build_navigation(self):
        return [
            [
                InlineKeyboardButton(
                    "💬 Free Chat (Pause)", callback_data="nav_free_chat"
                )
            ],
            [
                InlineKeyboardButton("🔁 Restart", callback_data="nav_restart"),
                InlineKeyboardButton("⬅️ Back", callback_data="nav_back"),
            ],
        ]

    def build_chat_navigation(self):
        return [
            [
                InlineKeyboardButton("🔁 Restart Wizard", callback_data="nav_restart"),
                InlineKeyboardButton("⬅️ Back to Wizard", callback_data="nav_back"),
            ]
        ]

    def build_topic_or_task_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("📝 Topic", callback_data="topic_or_task:topic"),
                InlineKeyboardButton("🎯 Task", callback_data="topic_or_task:task"),
            ],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    def build_length_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("📃 Short", callback_data="length:short"),
                InlineKeyboardButton("📖 Medium", callback_data="length:medium"),
                InlineKeyboardButton("📚 Long", callback_data="length:long"),
            ],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    def build_level_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("👶 Beginner", callback_data="level:beginner"),
                InlineKeyboardButton(
                    "📘 Intermediate", callback_data="level:intermediate"
                ),
                InlineKeyboardButton("🎓 Advanced", callback_data="level:advanced"),
            ],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    def build_info_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("💧 Low", callback_data="info:low"),
                InlineKeyboardButton("🌊 Medium", callback_data="info:medium"),
                InlineKeyboardButton("🌊🌊 High", callback_data="info:high"),
            ],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    def build_tone_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton(
                    "🏛️ Professional", callback_data="tone:professional"
                ),
                InlineKeyboardButton("😎 Casual", callback_data="tone:casual"),
                InlineKeyboardButton("😄 Friendly", callback_data="tone:friendly"),
            ],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    def build_confirm_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("✅ Confirm", callback_data="confirm:confirm")],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    def build_start_configuration_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton(
                    "▶️ Start configuration", callback_data="start_config"
                )
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_navigation_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(self.build_navigation())

    def build_website_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🚫 No website", callback_data="website:no")],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    def build_document_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🚫 No document", callback_data="document:no")],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    def build_additional_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton(
                    "🚫 No additional info", callback_data="additional:no"
                )
            ],
        ] + self.build_navigation()
        return InlineKeyboardMarkup(keyboard)

    # Helper

    def reset_wizard_data(self, context: CallbackContext) -> None:
        """resets all wizard data."""
        user_data = context.user_data
        for key in [
            "topic",
            "length",
            "language_level",
            "information",
            "language",
            "tone",
            "additional_information",
            "state_stack",
            "current_state",
        ]:
            user_data.pop(key, None)
        user_data["state_stack"] = []
        user_data["current_state"] = int(S.TOPIC_OR_TASK)
        user_data["history"] = []
        user_data["file_paths"] = []
        self.logger.info("Wizard data reset (restart).")

    def push_state(self, context: CallbackContext, from_state: S) -> None:
        """memorize prior state"""
        stack = context.user_data.setdefault("state_stack", [])
        stack.append(int(from_state))

    def clear_state_data(self, context: CallbackContext, state: S) -> None:
        """deletes state data"""
        user_data = context.user_data
        if state in (S.TOPIC, S.TASK):
            user_data.pop("topic", None)
        elif state == S.LENGTH:
            user_data.pop("length", None)
        elif state == S.LEVEL:
            user_data.pop("language_level", None)
        elif state == S.INFO:
            user_data.pop("information", None)
        elif state == S.LANGUAGE:
            user_data.pop("language", None)
        elif state == S.TONE:
            user_data.pop("tone", None)
        elif state == S.ADDITIONAL:
            user_data.pop("additional_information", None)

    def set_last_wizard_message(self, context: CallbackContext, message) -> None:
        """Store the last wizard message, to remove its buttons later on."""
        context.user_data["last_wizard_message"] = {
            "chat_id": message.chat_id,
            "message_id": message.message_id,
        }

    async def clear_last_wizard_keyboard(self, context: CallbackContext) -> None:
        """Remove inline keyboard from the last wizard message."""
        info = context.user_data.get("last_wizard_message")
        if not info:
            return

        try:
            await context.bot.edit_message_reply_markup(
                chat_id=info["chat_id"],
                message_id=info["message_id"],
                reply_markup=None,
            )
        except BadRequest as e:

            self.logger.debug(f"clear_last_wizard_keyboard: {e}")

    async def ask_state_question(
        self, update: Update, context: CallbackContext, state: S
    ) -> None:
        message = update.effective_message

        if state == S.TOPIC_OR_TASK:
            text = (
                "🔵⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪\n\n"
                "Topic or Task\n\n"
                "Let's configure your blog article! ✍️\n\n"
                "First, do you already have a <b>TOPIC</b> or rather a <b>TASK</b> "
                "the article should fulfil?"
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_topic_or_task_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.TOPIC:
            text = (
                "🔵🔵⚪⚪⚪⚪⚪⚪⚪⚪⚪\n\n"
                "Great, you've chosen <b>TOPIC</b>! 📝\n\n"
                "What topic should the blog article be about?"
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_navigation_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.TASK:
            text = (
                "🔵🔵⚪⚪⚪⚪⚪⚪⚪⚪⚪\n\n"
                "Great, you've chosen <b>TOPIC</b>! 🎯\n\n"
                "What task should the blog article fulfil? "
                "Please describe it in a short sentence."
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_navigation_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.WEBSITE:
            text = (
                "🔵🔵🔵⚪⚪⚪⚪⚪⚪⚪⚪\n\n"
                "Do you have a website with information that should be included?\n"
                "If yes, please send the URL.\n"
                "If not, tap the button below or type 'no'."
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_website_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.DOCUMENT:
            text = (
                "🔵🔵🔵🔵⚪⚪⚪⚪⚪⚪⚪\n\n"
                "Do you have a <b>DOCUMENT</b> (PDF, DOCX, TXT) with information to include?\n"
                "If yes, upload it now.\n"
                "If not, tap the button below or type 'no'."
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_document_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.LENGTH:
            text = (
                "🔵🔵🔵🔵🔵⚪⚪⚪⚪⚪⚪\n\n"
                "How long should the blog article be? Choose one of the options below 👇"
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_length_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.LEVEL:
            text = (
                "🔵🔵🔵🔵🔵🔵⚪⚪⚪⚪⚪\n\n"
                "What <b>LANGUAGE LEVEL</b> should it be? 👇"
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_level_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.INFO:
            text = (
                "🔵🔵🔵🔵🔵🔵🔵⚪⚪⚪⚪\n\n"
                "What <b>INFORMATION LEVEL</b> should it be? 👇"
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_info_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.LANGUAGE:
            text = (
                "🔵🔵🔵🔵🔵🔵🔵🔵⚪⚪⚪\n\n"
                "What <b>LANGUAGE</b> should the article be in? 🌍\n"
                "(e.g. English, German, Spanish)"
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_navigation_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.TONE:
            text = (
                "🔵🔵🔵🔵🔵🔵🔵🔵🔵⚪⚪\n\n"
                "What <b>TONE</b> should the article have? 🎨"
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_tone_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.ADDITIONAL:
            text = (
                "🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵⚪\n\n"
                "Do you have any <b>ADDITIONAL INFORMATION</b> you want to include?\n"
                "If not, tap the button below or type 'no'."
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_additional_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.CONFIRM:
            user_data = context.user_data
            text = (
                "🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵\n\n"
                "Thanks! Here's your configuration:\n\n"
                f"- Topic or Task: {user_data.get('topic')}\n"
                f"- Length: {user_data.get('length')}\n"
                f"- Language Level: {user_data.get('language_level')}\n"
                f"- Information Level: {user_data.get('information')}\n"
                f"- Language: {user_data.get('language')}\n"
                f"- Tone: {user_data.get('tone')}\n"
                f"- Additional Information: {user_data.get('additional_information')}\n\n"
                "If everything looks good, confirm to start generation."
            )
            sent = await message.reply_html(
                text, reply_markup=self.build_confirm_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.FREE_CHAT:
            text = (
                "💬 <b>Free Chat Mode</b>\n\n"
                "You paused the wizard to chat freely, to e.g. brainstorn, ask questions, or check RAG-Infos.\n\n"
                "Press <b>BACK TO WIZARD</b> to return to the last step."
            )

            sent = await message.reply_html(
                text, reply_markup=InlineKeyboardMarkup(self.build_chat_navigation())
            )
            self.set_last_wizard_message(context, sent)

        else:
            await message.reply_text("Unknown state. Please /start again.")

    async def go_to_state(
        self,
        update: Update,
        context: CallbackContext,
        from_state: S | None,
        to_state: S,
    ) -> int:
        """bridge between the states"""
        if from_state is not None:
            self.push_state(context, from_state)
        context.user_data["current_state"] = int(to_state)
        await self.ask_state_question(update, context, to_state)
        return int(to_state)

    # Chat

    async def chat(self, update: Update, context: CallbackContext):
        """Free LLM Chat"""
        response = ""
        try:
            self.logger.debug(f"chat: called with message {str(update.message.text)}")
            context.user_data["history"] = context.user_data.get("history", []) + [
                update.message.text
            ]
            history = "\n".join(context.user_data["history"])
            response = str(self.ai.invoke(history))
            await update.message.reply_html(response)
            self.logger.debug(f"chat: answered with {str(response)}")
            context.user_data["history"].append(str(response))
            return self.CHAT

        except BadRequest as b:
            if b.message == "Message is too long":
                responses = response.split("\n\n")
                self.logger.warning("chat: Message too long, split into smaller parts.")
                for r in responses:
                    await update.message.reply_text(r)
            return self.CHAT

        except Exception as e:
            await update.message.reply_text(f"chat: An error occurred: {str(e)}")
            return self.CHAT

    # introduction

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Entry: Explains bot and navigation."""
        try:
            user = update.effective_user
            message = update.effective_message

            self.logger.info(
                f"start: Conversation started with {str(user.mention_html())}"
            )
            text = (
                f"Hi {user.mention_html()}! 👋\n\n"
                "I'm your personal <b>AI Editorial Team</b>, powered by RAG and Multi-Agent Systems. 🤖✍️\n\n"
                "I'll guide you through a wizard to create high-quality blog articles:\n"
                "• <b>Content:</b> Define a Topic or specific Task\n"
                "• <b>Knowledge:</b> Add Sources (Websites 🔗 / Documents 📄)\n"
                "• <b>Style:</b> Set Length, Tone, Language & Information Level\n\n"
                "<b>🚀 Navigation Features:</b>\n"
                "You can control the process at any time:\n\n"
                "💬 <b>Free Chat (Pause)</b> - Switch to chat mode to brainstorm ideas or verify your uploaded documents. I'll keep the context for the article!\n"
                "⬅️ <b>Back</b> - Go one step back to adjust your previous answer\n"
                "🔁 <b>Restart</b> - Reset the wizard and start from scratch\n\n"
                "Ready to create content? Start the configuration below 👇"
            )
            await message.reply_html(
                text,
                reply_markup=self.build_start_configuration_keyboard(),
            )
            context.user_data.setdefault("history", [])
            return self.CHAT

        except Exception as e:
            await update.effective_message.reply_text(f"An error occurred: {str(e)}")
            self.logger.error(f"start: exception {str(e)}")
            return self.CHAT

    # start for wizard

    async def start_configuration_entry(self, update: Update, context: CallbackContext):
        """starts wizard."""
        try:

            self.reset_wizard_data(context)
            await self.ask_state_question(update, context, S.TOPIC_OR_TASK)
            return int(S.TOPIC_OR_TASK)

        except Exception as e:
            self.logger.error(f"start_configuration_entry: error {e}", exc_info=True)
            await update.effective_message.reply_text(
                "An unexpected error occurred while starting the configuration."
            )
            return ConversationHandler.END

    async def start_configuration_button(
        self, update: Update, context: CallbackContext
    ):
        query = update.callback_query
        await query.answer()
        self.logger.debug("start_configuration_button: pressed.")

        await query.edit_message_text(
            "Great, let's configure your blog article! ✍️\n\n"
            "I'll guide you through a few steps. You can always go back or restart using the navigation buttons."
        )

        return await self.start_configuration_entry(update, context)

    # Navigation: Restart / Back / Free Chat

    async def handle_navigation(self, update: Update, context: CallbackContext) -> int:
        """handles navigation with 'restart' & 'back' & 'chat'"""
        query = update.callback_query
        data = query.data
        await query.answer()

        await self.clear_last_wizard_keyboard(context)

        user_data = context.user_data
        current_state_val = user_data.get("current_state", int(S.TOPIC_OR_TASK))
        current_state = S(current_state_val)

        self.logger.debug(
            f"handle_navigation: data={data}, current_state={current_state}"
        )

        if data == "nav_restart":
            self.reset_wizard_data(context)
            await query.message.reply_text("Wizard restarted. 🔁")
            await self.ask_state_question(update, context, S.TOPIC_OR_TASK)
            return int(S.TOPIC_OR_TASK)

        if data == "nav_free_chat":
            self.logger.info("Switching to Free Chat mode via Navigation.")
            return await self.go_to_state(
                update, context, from_state=current_state, to_state=S.FREE_CHAT
            )

        if data == "nav_back":
            stack = user_data.get("state_stack", [])
            if not stack:
                self.logger.debug("handle_navigation: back at first state.")
                user_data["current_state"] = int(S.TOPIC_OR_TASK)
                await query.message.reply_text("You are already at the first step.")
                await self.ask_state_question(update, context, S.TOPIC_OR_TASK)
                return int(S.TOPIC_OR_TASK)

            if current_state != S.FREE_CHAT:
                self.clear_state_data(context, current_state)

            prev_state_val = stack.pop()
            prev_state = S(prev_state_val)
            user_data["state_stack"] = stack
            user_data["current_state"] = int(prev_state)

            self.logger.debug(
                f"handle_navigation: going back from {current_state} to {prev_state}"
            )

            if current_state == S.FREE_CHAT:
                await query.message.reply_text("Leaving Chat. Back to Wizard. ⬅️")
            else:
                await query.message.reply_text("Going back one step. ⬅️")

            await self.ask_state_question(update, context, prev_state)
            return int(prev_state)

        await query.message.reply_text("Unknown navigation action.")
        return current_state_val

    # Step: Topic or Task

    async def topic_or_task_button(
        self, update: Update, context: CallbackContext
    ) -> int:
        query = update.callback_query
        data = query.data
        await query.answer()

        _, choice = data.split(":", 1)
        self.logger.debug(f"topic_or_task_button: choice={choice}")

        base_question = (
            "🔵⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪\n\n"
            "Let's configure your blog article! ✍️\n\n"
            "First, do you already have a *topic* or rather a *task* "
            "the article should fulfil?"
        )
        await query.edit_message_text(
            f"{base_question}\n\n✅ Selected: {choice.capitalize()}"
        )

        if choice == "topic":
            return await self.go_to_state(
                update, context, from_state=S.TOPIC_OR_TASK, to_state=S.TOPIC
            )

        if choice == "task":
            return await self.go_to_state(
                update, context, from_state=S.TOPIC_OR_TASK, to_state=S.TASK
            )

        await query.message.reply_text(
            "Please choose either *Topic* or *Task* using the buttons."
        )
        return int(S.TOPIC_OR_TASK)

    async def topic_or_task(self, update: Update, context: CallbackContext) -> int:
        text = (update.message.text or "").strip().lower()
        self.logger.debug(f"topic_or_task {text}")

        if text == "topic":
            await self.clear_last_wizard_keyboard(context)
            return await self.go_to_state(
                update, context, from_state=S.TOPIC_OR_TASK, to_state=S.TOPIC
            )
        if text == "task":
            await self.clear_last_wizard_keyboard(context)
            return await self.go_to_state(
                update, context, from_state=S.TOPIC_OR_TASK, to_state=S.TASK
            )

        await update.message.reply_text(
            "Please choose *Topic* or *Task* using the buttons."
        )
        return int(S.TOPIC_OR_TASK)

    # Step: Topic / Task

    async def topic(self, update: Update, context: CallbackContext) -> int:
        """Saves the topic in the user data."""
        text = (update.message.text or "").strip()
        self.logger.debug(f"topic: {text}")
        context.user_data["topic"] = text

        await self.clear_last_wizard_keyboard(context)

        return await self.go_to_state(
            update, context, from_state=S.TOPIC, to_state=S.WEBSITE
        )

    async def task(self, update: Update, context: CallbackContext) -> int:
        """Saves the task in the user data."""
        text = (update.message.text or "").strip()
        self.logger.debug(f"task: {text}")
        context.user_data["topic"] = text

        await self.clear_last_wizard_keyboard(context)

        return await self.go_to_state(
            update, context, from_state=S.TASK, to_state=S.WEBSITE
        )

    # Step: Website

    async def website(self, update: Update, context: CallbackContext) -> int:
        """Processes website input or skip with 'no'."""
        message = update.message

        if message is None or (message.text or "").strip() == "":
            await update.effective_chat.send_message(
                "Please send a URL or type 'no' to skip this step."
            )
            return int(S.WEBSITE)

        text = (message.text or "").strip()
        self.logger.debug(f"website: {text}")

        if text.lower() == "no":
            self.logger.debug("website: user typed 'no'")

            await self.clear_last_wizard_keyboard(context)

            return await self.go_to_state(
                update, context, from_state=S.WEBSITE, to_state=S.DOCUMENT
            )

        if not (text.startswith("http://") or text.startswith("https://")):
            await message.reply_text(
                "That doesn't look like a valid URL.\n"
                "Please send a link starting with http:// or https://, or type 'no' to skip."
            )
            return int(S.WEBSITE)

        try:
            context.user_data.setdefault("file_paths", []).append(text)
            await message.reply_text(
                "✅ Got your website. I'll use it as an information source."
            )
            self.logger.info(f"Website added as RAG source: {text}")
        except Exception as e:
            self.logger.exception(f"Error while adding website RAG tool: {e}")
            await message.reply_text(
                "⚠️ I couldn't process this website as a source.\n"
                "I'll continue without it. You can still upload a document in the next step."
            )

        await self.clear_last_wizard_keyboard(context)

        return await self.go_to_state(
            update, context, from_state=S.WEBSITE, to_state=S.DOCUMENT
        )

    async def website_button(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        await query.answer()
        self.logger.debug("website_button: user chose 'No website'")

        base_question = (
            "🔵🔵🔵⚪⚪⚪⚪⚪⚪⚪⚪\n\n"
            "Do you have a website with information that should be included?\n"
            "If yes, please send the URL.\n"
            "If not, tap the button below or type 'no'."
        )

        await query.edit_message_text(f"{base_question}\n\n✅ Selected: No website")

        return await self.go_to_state(
            update, context, from_state=S.WEBSITE, to_state=S.DOCUMENT
        )

    # Step: Document

    async def document(self, update: Update, context: CallbackContext) -> int:
        """Processes document upload or skip with 'no'."""
        message = update.message
        text = (message.text or "").strip() if message and message.text else None
        document = message.document if message else None

        if text is not None and text.lower() == "no" and document is None:
            self.logger.debug("document: user typed 'no'")

            await self.clear_last_wizard_keyboard(context)

            return await self.go_to_state(
                update, context, from_state=S.DOCUMENT, to_state=S.LENGTH
            )

        if document is None and (text is None or text == ""):
            await message.reply_text(
                "Please upload a document (PDF, DOCX, TXT) or type 'no' to skip this step."
            )
            return int(S.DOCUMENT)

        if document:
            self.logger.debug(
                f"document: received document {document.file_name} ({document.mime_type})"
            )

            if document.mime_type not in self.VALID_MIME_TYPES:
                await message.reply_text(
                    f"Unsupported file type: {document.mime_type}.\n"
                    "Please upload PDF, DOCX or TXT, or type 'no'."
                )
                return int(S.DOCUMENT)

            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            file_path = str(DOCUMENTS_DIR / document.file_name)

            try:
                file = await context.bot.get_file(document.file_id)
                await file.download_to_drive(file_path)
                self.logger.info(f"Document saved to {file_path}")
            except Exception as e:
                self.logger.exception(f"Error downloading document: {e}")
                await message.reply_text(
                    "⚠️ I couldn't download your document.\n"
                    "Please try again, or type 'no' to skip this step."
                )
                return int(S.DOCUMENT)

            try:
                context.user_data.setdefault("file_paths", []).append(file_path)

                await message.reply_text(
                    "✅ Got your document. I'll use it as an information source."
                )
            except Exception as e:
                self.logger.exception(f"Error while adding document RAG tool: {e}")
                await message.reply_text(
                    "⚠️ I couldn't process this document as a source.\n"
                    "I'll continue without it."
                )

        await self.clear_last_wizard_keyboard(context)

        return await self.go_to_state(
            update, context, from_state=S.DOCUMENT, to_state=S.LENGTH
        )

    async def no_document_button(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        await query.answer()
        self.logger.debug("no_document_button: user chose 'No document'")

        base_question = (
            "🔵🔵🔵🔵⚪⚪⚪⚪⚪⚪⚪\n\n"
            "Do you have a *document* (PDF, DOCX, TXT) with information to include?\n"
            "If yes, upload it now.\n"
            "If not, tap the button below or type 'no'."
        )

        await query.edit_message_text(f"{base_question}\n\n✅ Selected: No document")

        return await self.go_to_state(
            update, context, from_state=S.DOCUMENT, to_state=S.LENGTH
        )

    # Step: Length

    async def length_button(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        data = query.data
        await query.answer()
        _, value = data.split(":", 1)
        self.logger.debug(f"length_button: {value}")
        context.user_data["length"] = value

        question_text = (
            "🔵🔵🔵🔵🔵⚪⚪⚪⚪⚪⚪\n\n"
            "How long should the blog article be? Choose one of the options below 👇"
        )
        await query.edit_message_text(
            f"{question_text}\n\n✅ Selected: {value.capitalize()}"
        )

        return await self.go_to_state(
            update, context, from_state=S.LENGTH, to_state=S.LEVEL
        )

    async def length_text(self, update: Update, context: CallbackContext) -> int:
        text = update.message.text
        self.logger.debug(f"length_text: {text}")
        context.user_data["length"] = text
        await self.clear_last_wizard_keyboard(context)
        return await self.go_to_state(
            update, context, from_state=S.LENGTH, to_state=S.LEVEL
        )

    # Step: Language Level

    async def language_level_button(
        self, update: Update, context: CallbackContext
    ) -> int:
        query = update.callback_query
        data = query.data
        await query.answer()
        _, value = data.split(":", 1)
        self.logger.debug(f"language_level_button: {value}")
        context.user_data["language_level"] = value

        question_text = (
            "🔵🔵🔵🔵🔵🔵⚪⚪⚪⚪⚪\n\n" "What *language level* should it be? 👇"
        )
        await query.edit_message_text(
            f"{question_text}\n\n✅ Selected: {value.capitalize()}"
        )

        return await self.go_to_state(
            update, context, from_state=S.LEVEL, to_state=S.INFO
        )

    async def language_level(self, update: Update, context: CallbackContext) -> int:
        text = update.message.text
        self.logger.debug(f"language_level: {text}")
        context.user_data["language_level"] = text
        await self.clear_last_wizard_keyboard(context)
        return await self.go_to_state(
            update, context, from_state=S.LEVEL, to_state=S.INFO
        )

    # Step: Information Level

    async def info_level_button(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        data = query.data
        await query.answer()
        _, value = data.split(":", 1)
        self.logger.debug(f"info_level_button: {value}")
        context.user_data["information"] = value

        question_text = (
            "🔵🔵🔵🔵🔵🔵🔵⚪⚪⚪⚪\n\n" "What *information level* should it be? 👇"
        )
        await query.edit_message_text(
            f"{question_text}\n\n✅ Selected: {value.capitalize()}"
        )

        return await self.go_to_state(
            update, context, from_state=S.INFO, to_state=S.LANGUAGE
        )

    async def info_level(self, update: Update, context: CallbackContext) -> int:
        text = update.message.text
        self.logger.debug(f"info_level: {text}")
        context.user_data["information"] = text
        await self.clear_last_wizard_keyboard(context)
        return await self.go_to_state(
            update, context, from_state=S.INFO, to_state=S.LANGUAGE
        )

    # Step: Language

    async def language(self, update: Update, context: CallbackContext) -> int:
        """Saves the language in the user data."""
        text = (update.message.text or "").strip()
        self.logger.debug(f"language: {text}")
        context.user_data["language"] = text
        await self.clear_last_wizard_keyboard(context)
        return await self.go_to_state(
            update, context, from_state=S.LANGUAGE, to_state=S.TONE
        )

    # Step: Tone

    async def tone_button(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        data = query.data
        await query.answer()
        _, value = data.split(":", 1)
        self.logger.debug(f"tone_button: {value}")
        context.user_data["tone"] = value

        question_text = (
            "🔵🔵🔵🔵🔵🔵🔵🔵🔵⚪⚪\n\nWhat *tone* should the article have? 🎨"
        )
        await query.edit_message_text(
            f"{question_text}\n\n✅ Selected: {value.capitalize()}"
        )

        return await self.go_to_state(
            update, context, from_state=S.TONE, to_state=S.ADDITIONAL
        )

    async def tone(self, update: Update, context: CallbackContext) -> int:
        text = update.message.text
        self.logger.debug(f"tone: {text}")
        context.user_data["tone"] = text
        await self.clear_last_wizard_keyboard(context)
        return await self.go_to_state(
            update, context, from_state=S.TONE, to_state=S.ADDITIONAL
        )

    # Step: Additional Information

    async def additional(self, update: Update, context: CallbackContext) -> int:
        """Processes additional text-info or 'no'."""
        text = (update.message.text or "").strip()
        self.logger.debug(f"additional: {text}")

        if text.lower() == "no":
            context.user_data["additional_information"] = ""
        else:
            context.user_data["additional_information"] = text

        await self.clear_last_wizard_keyboard(context)

        return await self.go_to_state(
            update, context, from_state=S.ADDITIONAL, to_state=S.CONFIRM
        )

    async def additional_no_button(
        self, update: Update, context: CallbackContext
    ) -> int:
        query = update.callback_query
        await query.answer()
        self.logger.debug("additional_no_button: user chose 'No additional info'")

        context.user_data["additional_information"] = ""

        base_question = (
            "🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵⚪\n\n"
            "Do you have any *additional information* you want to include?\n"
            "If not, tap the button below or type 'no'."
        )

        await query.edit_message_text(
            f"{base_question}\n\n✅ Selected: No additional info"
        )

        return await self.go_to_state(
            update, context, from_state=S.ADDITIONAL, to_state=S.CONFIRM
        )

    # Step: Confirm

    async def confirm_button(self, update: Update, context: CallbackContext) -> int:
        query = update.callback_query
        data = query.data
        await query.answer()
        _, action = data.split(":", 1)
        self.logger.debug(f"confirm_button: action={action}")

        if action != "confirm":
            await query.message.reply_text(
                "Please use the Confirm button to start the generation."
            )
            return int(S.CONFIRM)

        user_data = context.user_data
        summary_text = (
            "🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵\n\n"
            "Thanks! Here's your configuration:\n\n"
            f"- Topic or Task: {user_data.get('topic')}\n"
            f"- Length: {user_data.get('length')}\n"
            f"- Language Level: {user_data.get('language_level')}\n"
            f"- Information Level: {user_data.get('information')}\n"
            f"- Language: {user_data.get('language')}\n"
            f"- Tone: {user_data.get('tone')}\n"
            f"- Additional Information: {user_data.get('additional_information')}\n\n"
            "✅ Selected: Confirm"
        )

        await query.edit_message_text(summary_text)

        inputs = {
            "topic": user_data.get("topic"),
            "length": user_data.get("length"),
            "language_level": user_data.get("language_level"),
            "information_level": user_data.get("information"),
            "language": user_data.get("language"),
            "tone": user_data.get("tone"),
            "additional_information": user_data.get("additional_information"),
            "history": user_data.get("history", []),
        }

        try:
            file_paths = context.user_data.get("file_paths", [])
            if file_paths:
                await query.message.reply_text("📚 Indexing sources...")
                setup_vectorstore(file_paths)

            graph_inputs = {
                "topic": inputs.get("topic"),
                "target_len": inputs.get("length"),
                "language_level": inputs.get("language_level"),
                "information_level": inputs.get("information_level"),
                "language": inputs.get("language"),
                "tone": inputs.get("tone"),
                "additional_info": inputs.get("additional_information"),
                "source_documents": file_paths,
                "history": inputs.get("history", []),
                "research_data": [],
                "outline": [],
                "draft": "",
                "final_article": "",
                "revision_count": 0,
            }

            app = create_graph()

            status_text = "🚀 GENERATION STATUS\n\n"
            status_msg = await update.effective_message.reply_html(
                status_text + "⏳ 🕵️ Researcher is gathering information..."
            )

            final_state = graph_inputs

            async for output in app.astream(graph_inputs):
                for node_name, state_update in output.items():
                    final_state.update(state_update)

                    if node_name == "researcher":
                        status_text += "✅ 🕵️ Researcher finished.\n"
                        await status_msg.edit_text(
                            status_text + "⏳ 🏗️ Editor is creating the outline..."
                        )

                    elif node_name == "editor":
                        status_text += "✅ 🏗️ Editor finished.\n"
                        await status_msg.edit_text(
                            status_text + "⏳ ✍️ Writer is drafting the article..."
                        )

                    elif node_name == "writer":
                        status_text += "✅ ✍️ Writer finished.\n"
                        await status_msg.edit_text(
                            status_text + "⏳ ⚖️ Fact Checker is verifying facts..."
                        )

                    elif node_name == "fact_checker":
                        status_text += "✅ ⚖️ Fact Checker finished.\n"
                        critique = state_update.get("critique", "").strip().upper()
                        rev_count = state_update.get("revision_count", 0)

                        if (
                            critique == "PASS"
                            or critique.startswith("PASS")
                            or critique == ""
                            or rev_count >= 2
                        ):
                            await status_msg.edit_text(
                                status_text
                                + "⏳ ✨ Polisher is formatting the final text..."
                            )
                        else:
                            await status_msg.edit_text(
                                status_text
                                + f"⚠️ ✍️ Fact Checker found errors! Writer is rewriting (Revision {rev_count})..."
                            )

                    elif node_name == "polisher":
                        status_text += "✅ ✨ Polisher finished.\n"
                        await status_msg.edit_text(
                            status_text + "🎉 Generation complete!"
                        )

            final_text = final_state.get("final_article", "⚠️ No article generated.")

            article_title = inputs.get("topic") or "Article"
            await self.send_file_response(update, final_text, article_title)

            self.logger.debug("confirm_button: Graph run successful.")

        except Exception as e:
            self.logger.exception(f"confirm_button: error during crew run: {e}")
            await query.message.reply_text(
                "⚠️ An error occurred while generating the article. Please try again."
            )

        return ConversationHandler.END

    async def confirm(self, update: Update, context: CallbackContext) -> int:
        text = (update.message.text or "").strip().lower()
        self.logger.debug(f"confirm: {text}")

        if text in ("yes", "y", "ja"):

            inputs = {
                "topic": context.user_data.get("topic"),
                "length": context.user_data.get("length"),
                "language_level": context.user_data.get("language_level"),
                "information_level": context.user_data.get("information"),
                "language": context.user_data.get("language"),
                "tone": context.user_data.get("tone"),
                "additional_information": context.user_data.get(
                    "additional_information"
                ),
                "history": context.user_data.get("history", []),
            }

            try:

                file_paths = context.user_data.get("file_paths", [])
                if file_paths:
                    await update.message.reply_text("📚 Indexing sources...")
                    setup_vectorstore(file_paths)

                graph_inputs = {
                    "topic": inputs.get("topic"),
                    "target_len": inputs.get("length"),
                    "language_level": inputs.get("language_level"),
                    "information_level": inputs.get("information_level"),
                    "language": inputs.get("language"),
                    "tone": inputs.get("tone"),
                    "additional_info": inputs.get("additional_information"),
                    "source_documents": file_paths,
                    "history": inputs.get("history", []),
                    "research_data": [],
                    "outline": [],
                    "draft": "",
                    "final_article": "",
                    "revision_count": 0,
                }

                app = create_graph()
                status_text = "🚀 GENERATION STATUS\n\n"
                status_msg = await update.effective_message.reply_html(
                    status_text + "⏳ 🕵️ Researcher is gathering information..."
                )

                final_state = graph_inputs

                async for output in app.astream(graph_inputs):
                    for node_name, state_update in output.items():
                        final_state.update(state_update)

                        if node_name == "researcher":
                            status_text += "✅ 🕵️ Researcher finished.\n"
                            await status_msg.edit_text(
                                status_text + "⏳ 🏗️ Editor is creating the outline..."
                            )

                        elif node_name == "editor":
                            status_text += "✅ 🏗️ Editor finished.\n"
                            await status_msg.edit_text(
                                status_text + "⏳ ✍️ Writer is drafting the article..."
                            )

                        elif node_name == "writer":
                            status_text += "✅ ✍️ Writer finished.\n"
                            await status_msg.edit_text(
                                status_text + "⏳ ⚖️ Fact Checker is verifying facts..."
                            )

                        elif node_name == "fact_checker":
                            status_text += "✅ ⚖️ Fact Checker finished.\n"
                            critique = state_update.get("critique", "").strip().upper()
                            rev_count = state_update.get("revision_count", 0)

                            if (
                                critique == "PASS"
                                or critique.startswith("PASS")
                                or critique == ""
                                or rev_count >= 2
                            ):
                                await status_msg.edit_text(
                                    status_text
                                    + "⏳ ✨ Polisher is formatting the final text..."
                                )
                            else:
                                await status_msg.edit_text(
                                    status_text
                                    + f"⚠️ ✍️ Fact Checker found errors! Writer is rewriting (Revision {rev_count})..."
                                )

                        elif node_name == "polisher":
                            status_text += "✅ ✨ Polisher finished.\n"
                            await status_msg.edit_text(
                                status_text + "🎉 Generation complete!"
                            )

                final_text = final_state.get("final_article", "⚠️ No article generated.")

                article_title = inputs.get("topic") or "Article"
                await self.send_file_response(update, final_text, article_title)
                self.logger.debug("confirm: Graph run successful.")

            except Exception as e:
                self.logger.error(f"confirm: graph error {e}", exc_info=True)
                await update.message.reply_text(
                    "❌ An error occurred during article generation. Please try again."
                )

            return ConversationHandler.END

        if text in ("no", "n", "nein"):

            self.reset_wizard_data(context)
            await update.message.reply_text(
                "Configuration discarded. Restarting wizard. 🔁"
            )
            await self.ask_state_question(update, context, S.TOPIC_OR_TASK)
            return int(S.TOPIC_OR_TASK)

        await update.message.reply_text(
            "Please reply with 'yes' to confirm or 'no' to restart.\n"
            "Or use the buttons above."
        )
        return int(S.CONFIRM)

    async def free_chat_state(self, update: Update, context: CallbackContext) -> int:
        """processes messages, while in 'free chat'-mode."""
        user_text = update.message.text
        self.logger.debug(f"free_chat_state: received '{user_text}'")
        await self.clear_last_wizard_keyboard(context)

        context.user_data["history"] = context.user_data.get("history", []) + [
            user_text
        ]

        try:
            history_str = "\n".join(context.user_data["history"])

            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action="typing"
            )

            response = str(self.ai.invoke(history_str))
            context.user_data["history"].append(response)

            sent = await update.message.reply_text(
                response,
                reply_markup=InlineKeyboardMarkup(self.build_chat_navigation()),
            )

            self.set_last_wizard_message(context, sent)

        except Exception as e:
            self.logger.error(f"free_chat_state error: {e}")
            await update.message.reply_text(f"Error in chat: {e}")

        return int(S.FREE_CHAT)

    async def send_file_response(self, update: Update, content: str, topic: str):
        """
        creates md.-file of the finished blog-post, hands it to the user and deletes data afterwards.
        """
        safe_topic = "".join(
            c for c in topic if c.isalnum() or c in (" ", "_", "-")
        ).strip()
        safe_topic = safe_topic.replace(" ", "_")

        if not safe_topic:
            safe_topic = "Blog_Draft"

        filename = f"{safe_topic}.md"
        file_path = DOCUMENTS_DIR / filename

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            await update.effective_message.reply_text(
                "✅ FINISHED! Here is your draft as md.-file:"
            )

            await update.effective_message.reply_document(
                document=file_path, filename=filename, caption=f"📄 Draft: {safe_topic}"
            )
            self.logger.info(f"File sent successfully: {filename}")

        except Exception as e:
            self.logger.error(f"Failed to send file: {e}")
            await update.effective_message.reply_text(
                "⚠️ Failed to send file. Trying as blank text..."
            )

            await update.effective_message.reply_text(content[:4000])

        finally:

            if file_path.exists():
                try:
                    os.remove(file_path)
                except Exception as e:
                    self.logger.warning(f"Could not delete temp file {file_path}: {e}")

    # Start bot

    def start_bot(self) -> None:
        """Builds and starts the Telegram bot with the conversation handler."""
        application = Application.builder().token(self.token).build()

        conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler("start_configuration", self.start_configuration_entry),
                CallbackQueryHandler(
                    self.start_configuration_button,
                    pattern="^start_config$",
                ),
            ],
            states={
                S.FREE_CHAT: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND, self.free_chat_state
                    ),
                ],
                S.TOPIC_OR_TASK: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(
                        self.topic_or_task_button, pattern="^topic_or_task:"
                    ),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.topic_or_task,
                    ),
                ],
                S.TOPIC: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.topic,
                    ),
                ],
                S.TASK: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.task,
                    ),
                ],
                S.WEBSITE: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(self.website_button, pattern="^website:no$"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.website,
                    ),
                ],
                S.DOCUMENT: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(
                        self.no_document_button, pattern="^document:no$"
                    ),
                    MessageHandler(
                        filters.Document.ALL & ~filters.COMMAND,
                        self.document,
                    ),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.document,
                    ),
                ],
                S.LENGTH: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(self.length_button, pattern="^length:"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.length_text,
                    ),
                ],
                S.LEVEL: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(self.language_level_button, pattern="^level:"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.language_level,
                    ),
                ],
                S.INFO: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(self.info_level_button, pattern="^info:"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.info_level,
                    ),
                ],
                S.LANGUAGE: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.language,
                    ),
                ],
                S.TONE: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(self.tone_button, pattern="^tone:"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.tone,
                    ),
                ],
                S.ADDITIONAL: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(
                        self.additional_no_button, pattern="^additional:no$"
                    ),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.additional,
                    ),
                ],
                S.CONFIRM: [
                    CallbackQueryHandler(self.handle_navigation, pattern="^nav_"),
                    CallbackQueryHandler(self.confirm_button, pattern="^confirm:"),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.confirm,
                    ),
                ],
            },
            fallbacks=[],
            name="blog_config_conversation",
        )

        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("chat", self.chat))
        application.add_handler(conv_handler)
        application.run_polling()
