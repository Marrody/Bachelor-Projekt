import os
import shutil
import yaml

from crewai_tools.tools import (
    DOCXSearchTool,
    PDFSearchTool,
    TXTSearchTool,
    WebsiteSearchTool,
)
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
from src.ba_ragmas_chatbot import logger_config
from src.ba_ragmas_chatbot.crew import BaRagmasChatbot
from ba_ragmas_chatbot.states import S


class TelegramBot:

    CHAT = -1

    VALID_MIME_TYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_dir, "config", "configs.yaml")
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    token = config["chatbot_token"]["token"]
    llm_name = config["chatbot"]["llm"]["name"]
    llm_provider = config["chatbot"]["llm"]["provider_name"]
    llm_url = config["chatbot"]["llm"]["url"]
    embed_model_name = config["chatbot"]["embedding_model"]["name"]
    embed_model_provider = config["chatbot"]["embedding_model"]["provider_name"]
    embed_model_url = config["chatbot"]["embedding_model"]["url"]

    tools = []
    ai = OllamaLLM(model=llm_name)
    logger = logger_config.get_logger("telegram bot")

    # Utility – clear db

    def clear_db(self):
        """Deletes all database files related to ChromaDB."""
        db_folder = "./db"
        if os.path.exists(db_folder):
            shutil.rmtree(db_folder)
        os.makedirs(db_folder)

    # Helper – navigation + keyboards

    def build_navigation(self) -> InlineKeyboardMarkup:

        return [
            InlineKeyboardButton("🔁 Restart", callback_data="nav_restart"),
            InlineKeyboardButton("⬅️ Back", callback_data="nav_back"),
        ]

    def build_topic_or_task_keyboard(self) -> InlineKeyboardMarkup:

        keyboard = [
            [
                InlineKeyboardButton("📝 Topic", callback_data="topic_or_task:topic"),
                InlineKeyboardButton("🎯 Task", callback_data="topic_or_task:task"),
            ],
            self.build_navigation(),
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_length_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("📃 Short", callback_data="length:short"),
                InlineKeyboardButton("📖 Medium", callback_data="length:medium"),
                InlineKeyboardButton("📚 Long", callback_data="length:long"),
            ],
            self.build_navigation(),
        ]
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
            self.build_navigation(),
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_info_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("💧 Low", callback_data="info:low"),
                InlineKeyboardButton("🌊 Medium", callback_data="info:medium"),
                InlineKeyboardButton("🌊🌊 High", callback_data="info:high"),
            ],
            self.build_navigation(),
        ]
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
            self.build_navigation(),
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_confirm_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("✅ Confirm", callback_data="confirm:confirm")],
            self.build_navigation(),
        ]
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
        keyboard = [
            self.build_navigation(),
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_website_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🚫 No website", callback_data="website:no")],
            self.build_navigation(),
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_document_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🚫 No document", callback_data="document:no")],
            self.build_navigation(),
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_additional_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton(
                    "🚫 No additional info", callback_data="additional:no"
                )
            ],
            self.build_navigation(),
        ]
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
        self.tools.clear()
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
        """State-Handler: Asks the question for the given state and saves the message."""
        message = update.effective_message

        if state == S.TOPIC_OR_TASK:
            text = (
                "Let's configure your blog article! ✍️\n\n"
                "First, do you already have a *topic* or rather a *task* "
                "the article should fulfil?"
            )
            sent = await message.reply_text(
                text, reply_markup=self.build_topic_or_task_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.TOPIC:
            text = (
                "Great, you've chosen *Topic*! 📝\n\n"
                "What topic should the blog article be about?"
            )
            sent = await message.reply_text(
                text, reply_markup=self.build_navigation_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.TASK:
            text = (
                "Great, you've chosen *Task*! 🎯\n\n"
                "What task should the blog article fulfil? "
                "Please describe it in a short sentence."
            )
            sent = await message.reply_text(
                text, reply_markup=self.build_navigation_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.WEBSITE:
            text = (
                "Do you have a website with information that should be included?\n"
                "If yes, please send the URL.\n"
                "If not, tap the button below or type 'no'."
            )
            sent = await message.reply_text(
                text, reply_markup=self.build_website_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.DOCUMENT:
            text = (
                "Do you have a *document* (PDF, DOCX, TXT) with information to include?\n"
                "If yes, upload it now.\n"
                "If not, tap the button below or type 'no'."
            )
            sent = await message.reply_text(
                text, reply_markup=self.build_document_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.LENGTH:
            text = "How long should the blog article be? Choose one of the options below 👇"
            sent = await message.reply_text(
                text, reply_markup=self.build_length_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.LEVEL:
            text = "What *language level* should it be? 👇"
            sent = await message.reply_text(
                text, reply_markup=self.build_level_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.INFO:
            text = "What *information level* should it be? 👇"
            sent = await message.reply_text(
                text, reply_markup=self.build_info_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.LANGUAGE:
            text = (
                "What *language* should the article be in? 🌍\n"
                "(e.g. English, German, Spanish)"
            )
            sent = await message.reply_text(
                text, reply_markup=self.build_navigation_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.TONE:
            text = "What *tone* should the article have? 🎨"
            sent = await message.reply_text(
                text, reply_markup=self.build_tone_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.ADDITIONAL:
            text = (
                "Do you have any *additional information* you want to include?\n"
                "If not, tap the button below or type 'no'."
            )
            sent = await message.reply_text(
                text, reply_markup=self.build_additional_keyboard()
            )
            self.set_last_wizard_message(context, sent)

        elif state == S.CONFIRM:
            user_data = context.user_data
            text = (
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
            sent = await message.reply_text(
                text, reply_markup=self.build_confirm_keyboard()
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

    # Chat / Commands

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

    # /start – introduction

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
                "I'm a chatbot for creating blog articles using RAG and Multi-Agent Systems.\n\n"
                "We'll go through a short wizard where you configure:\n"
                "• Topic or Task\n"
                "• Sources (Websites / Documents)\n"
                "• Length, language level, info level\n"
                "• Language, tone, additional information\n\n"
                "At any time you can use:\n"
                "🔁 Restart – to reset the wizard and start from the beginning\n"
                "⬅️ Back – to go one step back and adjust your previous answer\n\n"
                "Ready? Start the configuration below 👇"
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

    # /help – deprecated

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            message = update.effective_message
            text = (
                "Help – RAG-MAS Blog Article Wizard 🤖\n\n"
                "Commands:\n"
                "• /start – Show introduction and start button\n"
                "• /chat – Free chat with the LLM\n\n"
                "During configuration you can always use:\n"
                "🔁 Restart – reset wizard and start from the beginning\n"
                "⬅️ Back – go one step back\n"
            )
            await message.reply_text(text)
            self.logger.debug("help: sent.")
            return self.CHAT

        except Exception as e:
            await update.effective_message.reply_text(f"An error occurred: {str(e)}")
            self.logger.error(f"help: exception {str(e)}")
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

    # Navigation: Restart & Back

    async def handle_navigation(self, update: Update, context: CallbackContext) -> int:
        """handles navigation with 'restart' & 'back'"""
        query = update.callback_query
        data = query.data
        await query.answer()

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

        if data == "nav_back":
            stack = user_data.get("state_stack", [])
            if not stack:

                self.logger.debug("handle_navigation: back at first state.")
                user_data["current_state"] = int(S.TOPIC_OR_TASK)
                await query.message.reply_text("You are already at the first step.")
                await self.ask_state_question(update, context, S.TOPIC_OR_TASK)
                return int(S.TOPIC_OR_TASK)

            self.clear_state_data(context, current_state)

            prev_state_val = stack.pop()
            prev_state = S(prev_state_val)
            user_data["state_stack"] = stack
            user_data["current_state"] = int(prev_state)

            self.logger.debug(
                f"handle_navigation: going back from {current_state} to {prev_state}"
            )
            await query.message.reply_text("Going back one step. ⬅️")
            await self.ask_state_question(update, context, prev_state)
            return int(prev_state)

        await query.message.reply_text(
            "Unknown navigation action. Please use the wizard buttons again."
        )
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
            return await self.go_to_state(
                update, context, from_state=S.TOPIC_OR_TASK, to_state=S.TOPIC
            )
        if text == "task":
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
            self.addWebsite(text)
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

            base_dir = os.path.dirname(__file__)
            documents_dir = os.path.join(base_dir, "documents")
            os.makedirs(documents_dir, exist_ok=True)

            file_path = os.path.join(documents_dir, document.file_name)

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
                if document.mime_type == "application/pdf":
                    self.addPDF(file_path)
                elif (
                    document.mime_type
                    == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ):
                    self.addDOCX(file_path)
                elif document.mime_type == "text/plain":
                    self.addTxt(file_path)

                await message.reply_text(
                    "✅ Got your document. I'll use it as an information source."
                )
                self.logger.info(
                    f"Document added as RAG source: {document.file_name} ({document.mime_type})"
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

        question_text = "What *language level* should it be? 👇"
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

        question_text = "What *information level* should it be? 👇"
        await query.edit_message_text(
            f"{question_text}\n\n✅ Selected: {value.capitalize()}"
        )

        return await self.go_to_state(
            update, context, from_state=S.INFO, to_state=S.LANGUAGE
        )

    async def info_level(self, update: Update, context: CallbackContext) -> int:
        text = update.message.text
        self.logger.debug(f"info_level: {text}")
        context.user_data["info_level"] = text
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

        question_text = "What *tone* should the article have? 🎨"
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
        await query.message.reply_text("Generating your article, please wait... ✅")

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
            bot = BaRagmasChatbot(self.tools)
            result = bot.crew().kickoff(inputs=inputs)
            await query.message.reply_text(str(result))
            self.logger.debug("confirm_button: Crew run successful.")
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

            await update.message.reply_text(
                "Generating your article, please wait... ✅"
            )

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
                bot = BaRagmasChatbot(self.tools)
                result = bot.crew().kickoff(inputs=inputs)
                await update.message.reply_text(str(result))
                self.logger.debug("confirm: Crew run successful.")
            except Exception as e:
                self.logger.error(f"confirm: crew error {e}", exc_info=True)
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
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(CommandHandler("chat", self.chat))
        application.add_handler(conv_handler)
        application.run_polling()

    # RAG-Tools

    def addWebsite(self, url):
        self.tools.append(
            WebsiteSearchTool(
                website=url,
                config=dict(
                    llm=dict(
                        provider=self.llm_provider,
                        config=dict(
                            model=self.llm_name,
                            base_url=self.llm_url,
                        ),
                    ),
                    embedder=dict(
                        provider=self.embed_model_provider,
                        config=dict(
                            model=self.embed_model_name,
                            base_url=self.embed_model_url,
                        ),
                    ),
                ),
            )
        )
        self.logger.info(f"Website-RAG-Tool added: {url}")

    def addPDF(self, location):
        self.tools.append(
            PDFSearchTool(
                pdf=location,
                config=dict(
                    llm=dict(
                        provider=self.llm_provider,
                        config=dict(
                            model=self.llm_name,
                            base_url=self.llm_url,
                        ),
                    ),
                    embedder=dict(
                        provider=self.embed_model_provider,
                        config=dict(
                            model=self.embed_model_name,
                            base_url=self.embed_model_url,
                        ),
                    ),
                ),
            )
        )
        self.logger.info(f"PDF-RAG-Tool added: {location}")

    def addDOCX(self, location):
        self.tools.append(
            DOCXSearchTool(
                docx=location,
                config=dict(
                    llm=dict(
                        provider=self.llm_provider,
                        config=dict(
                            model=self.llm_name,
                            base_url=self.llm_url,
                        ),
                    ),
                    embedder=dict(
                        provider=self.embed_model_provider,
                        config=dict(
                            model=self.embed_model_name,
                            base_url=self.embed_model_url,
                        ),
                    ),
                ),
            )
        )
        self.logger.info(f"DOCX-RAG-Tool added: {location}")

    def addTxt(self, location):
        self.tools.append(
            TXTSearchTool(
                txt=location,
                config=dict(
                    llm=dict(
                        provider=self.llm_provider,
                        config=dict(
                            model=self.llm_name,
                            base_url=self.llm_url,
                        ),
                    ),
                    embedder=dict(
                        provider=self.embed_model_provider,
                        config=dict(
                            model=self.embed_model_name,
                            base_url=self.embed_model_url,
                        ),
                    ),
                ),
            )
        )
        self.logger.info(f"TXT-RAG-Tool added: {location}")
