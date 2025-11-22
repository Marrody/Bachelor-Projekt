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
    KeyboardButton,
    ReplyKeyboardMarkup,
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
from ba_ragmas_chatbot.states import (
    S,
    first_state,
)


class TelegramBot:
    (
        CHAT,
        TOPIC,
        TASK,
        TOPIC_OR_TASK,
        WEBSITE,
        DOCUMENT,
        LENGTH,
        LANGUAGE_LEVEL,
        INFORMATION,
        LANGUAGE,
        TONE,
        CONFIRM,
        ADDITIONAL,
    ) = range(13)

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
    retry = False

    # Utility: clear DB

    def clear_db(self):
        """Deletes all database files related to ChromaDB."""
        db_folder = "./db"
        if os.path.exists(db_folder):
            shutil.rmtree(db_folder)
        os.makedirs(db_folder)

    # Helper: Keyboards

    def build_topic_or_task_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("📝 Topic", callback_data="topic"),
                InlineKeyboardButton("🎯 Task", callback_data="task"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_length_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("📃 Short", callback_data="short"),
                InlineKeyboardButton("📖 Medium", callback_data="medium"),
                InlineKeyboardButton("📚 Long", callback_data="long"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_level_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("👶 Beginner", callback_data="beginner"),
                InlineKeyboardButton("📘 Intermediate", callback_data="intermediate"),
                InlineKeyboardButton("🎓 Advanced", callback_data="advanced"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_info_keyboard(self) -> InlineKeyboardMarkup:

        keyboard = [
            [
                InlineKeyboardButton("💧 Low", callback_data="low"),
                InlineKeyboardButton("🌊 Medium", callback_data="medium_info"),
                InlineKeyboardButton("🌊🌊 High", callback_data="high"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_tone_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("🏛️ Professional", callback_data="professional"),
                InlineKeyboardButton("😎 Casual", callback_data="casual"),
                InlineKeyboardButton("😄 Friendly", callback_data="friendly"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_additional_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton(
                    "🚫 No additional info", callback_data="no_additional"
                )
            ],
            [InlineKeyboardButton("➕ Add details", callback_data="add_additional")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_website_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🚫 No website", callback_data="no_website")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_document_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("🚫 No document", callback_data="no_document_btn")],
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_confirm_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [
                InlineKeyboardButton("✅ Confirm", callback_data="confirm"),
                InlineKeyboardButton("🔁 Restart", callback_data="restart"),
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def build_start_keyboard(self) -> ReplyKeyboardMarkup:
        keyboard = [
            [KeyboardButton("/start_configuration")],
            [KeyboardButton("/help")],
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    # Chat / Commands

    async def chat(self, update: Update, context: CallbackContext):
        """Free LLM chat."""
        response = ""
        try:
            self.logger.debug(
                f"chat: Function successfully called with message {str(update.message.text)}"
            )
            context.user_data["history"] = context.user_data.get("history", []) + [
                update.message.text
            ]
            history = "\n".join(context.user_data["history"])
            response = str(self.ai.invoke(history))
            await update.message.reply_html(response)
            self.logger.debug(f"chat: Query successfully answered with {str(response)}")
            context.user_data["history"].append(str(response))
            return self.CHAT

        except BadRequest as b:
            if b.message == "Message is too long":
                responses = response.split("\n\n")
                self.logger.warning(
                    "chat: Message is too long, split up into small packets by double line."
                )
                for response in responses:
                    await update.message.reply_text(response)
            return self.CHAT

        except Exception as e:
            await update.message.reply_text(f"chat: An error occurred: {str(e)}")
            return self.CHAT

    async def clear(self, update: Update, context: CallbackContext):
        """Clears the conversation and user history, and returns to chat."""
        try:
            context.user_data.clear()
            context.user_data["history"] = []
            self.tools.clear()
            self.logger.info("clear: Conversation successfully cleared.")
            await update.message.reply_text(
                "Conversation successfully cleared! "
                "You can /start_configuration again or tap the buttons below.",
                reply_markup=self.build_start_keyboard(),
            )
            return self.CHAT

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. To re-clear the conversation, please send /clear again."
            )
            self.logger.error(
                f"clear: Tried to clear conversation, but an exception occurred: {str(e)}"
            )
            return self.CHAT

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        try:
            user = update.effective_user
            self.logger.info(
                f"start: Conversation successfully started with user {str(user.mention_html())}. "
            )
            response = (
                f"Hi {user.mention_html()}! This is a chatbot for creating blog articles using RAG and MAS. ✍️🤖\n\n"
                f"You can:\n"
                f"• /start_configuration – start the blog article configuration wizard\n"
                f"• /help – see an overview of all commands\n\n"
            )
            await update.message.reply_html(
                response,
                reply_markup=self.build_start_keyboard(),
            )
            context.user_data["history"] = []
            self.logger.debug("start: Response message successfully sent.")
            return self.CHAT

        except Exception as e:
            await update.message.reply_text(f"An error occurred: {str(e)}")
            self.logger.error(
                f"start: Tried to start conversation, but an exception occurred: {str(e)}"
            )
            return self.CHAT

    async def start_configuration(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Starts the blog article configuration process (entry point)."""

        try:
            self.logger.debug(
                "start_configuration: Blog article configuration started."
            )

            text = (
                "Great, let's configure your blog article! ✍️\n\n"
                "First: Do you already have a *topic* or rather a *task* the article should fulfil?"
            )
            await update.message.reply_text(
                text, reply_markup=self.build_topic_or_task_keyboard()
            )

            return S.TOPIC_OR_TASK

        except Exception as e:
            self.logger.error(f"Error in start_configuration: {e}", exc_info=True)
            await update.message.reply_text(
                "An unexpected error occurred while starting the configuration."
            )
            return ConversationHandler.END

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text(
                "Welcome to the RAG-MAS Blog Article Generator Bot! 📚\n\n"
                "Commands:\n"
                "• /start – Restart the conversation\n"
                "• /start_configuration – Start the blog article configuration wizard\n"
                "• /clear – Clear the conversation and user history\n"
                "• /cancel – End the conversation\n\n"
                "You can also use the buttons to navigate."
            )
            self.logger.debug("help: help sent.")
            return self.CHAT

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. To get help, please send /help again."
            )
            self.logger.error(
                f"help: Tried to respond with help, but an exception occurred: {str(e)}"
            )
            return self.CHAT

    # Step: Topic or Task

    async def topic_or_task_button(self, update: Update, context: CallbackContext):
        """Handles the InlineKeyboard choice 'topic' vs 'task'."""
        query = update.callback_query
        choice = query.data
        await query.answer()

        self.logger.debug(f"topic_or_task_button: User chose {choice}")

        base_text = (
            "Do you want to configure the article based on a *topic* or a *task*?\n\n"
            f"✅ Selected: {choice.capitalize()}"
        )
        await query.edit_message_text(base_text)

        if choice == "topic":
            response = (
                "Okay, topic it is! 📝\n\nWhat topic should the blog article be about?"
            )
            await query.message.reply_text(response)
            return self.TOPIC

        if choice == "task":
            response = (
                "Okay, task it is! 🎯\n\nWhat task should the blog article fulfil? "
                "Please describe it in a short sentence."
            )
            await query.message.reply_text(response)
            return self.TASK

        await query.message.reply_text(
            "Please choose whether you have a *topic* or a *task*."
        )
        return self.TOPIC_OR_TASK

    async def topic_or_task(self, update: Update, context: CallbackContext):
        """Fallback: if user still types 'topic' or 'task' as text."""
        try:
            text = (update.message.text or "").strip().lower()
            self.logger.debug(
                f"topic_or_task (text): called with message {update.message.text}"
            )

            if text == "topic":
                response = (
                    "Okay, topic it is! 📝 What topic should the blog article be about?"
                )
                await update.message.reply_text(response)
                return self.TOPIC

            if text == "task":
                response = (
                    "Okay, task it is! 🎯 What task should the blog article fulfil?"
                )
                await update.message.reply_text(response)
                return self.TASK

            response = "Please choose *Topic* or *Task* using the buttons below."
            await update.message.reply_text(
                response, reply_markup=self.build_topic_or_task_keyboard()
            )
            return self.TOPIC_OR_TASK

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease answer with 'topic' or 'task' again."
            )
            self.logger.error(f"topic_or_task: An exception occurred: {str(e)}")
            return self.TOPIC_OR_TASK

    # Step: Topic / Task

    async def topic(self, update: Update, context: CallbackContext):
        """Saves the topic in the user data"""
        try:
            self.logger.debug(
                f"topic: Function successfully called with message {str(update.message.text)}"
            )
            context.user_data["topic"] = update.message.text
            response = (
                "Great! 🌐 Do you have a link to a website with information to include?\n"
                "If yes, please reply with the link. If not, tap the button below or send 'no'."
            )
            await update.message.reply_text(
                response, reply_markup=self.build_website_keyboard()
            )
            return self.WEBSITE

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your topic."
            )
            self.logger.error(f"topic: An exception occurred: {str(e)}")
            return self.TOPIC

    async def task(self, update: Update, context: CallbackContext):
        """Saves the task in the user data"""
        try:
            self.logger.debug(
                f"task: Function successfully called with message {str(update.message.text)}"
            )
            context.user_data["topic"] = update.message.text
            response = (
                "Great! 🌐 Do you have a link to a website with information to include?\n"
                "If yes, please reply with the link. If not, tap the button below or send 'no'."
            )
            await update.message.reply_text(
                response, reply_markup=self.build_website_keyboard()
            )
            return self.WEBSITE

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your task."
            )
            self.logger.error(f"task: An exception occurred: {str(e)}")
            return self.TASK

    # Step: Website

    async def website_button(self, update: Update, context: CallbackContext):
        """Handles 'No website' button."""
        query = update.callback_query
        choice = query.data
        await query.answer()
        self.logger.debug(f"website_button: User chose {choice}")

        if choice == "no_website":
            await query.edit_message_text(
                "Do you have a website link to include?\n\n✅ Selected: No website"
            )
            response = (
                "Great! 📎 Do you have a *document* with information to include?\n"
                "If yes, upload the document now (PDF, DOCX, TXT). "
                "If not, tap the button below or send 'no'."
            )
            await query.message.reply_text(
                response, reply_markup=self.build_document_keyboard()
            )
            return self.DOCUMENT

        await query.message.reply_text(
            "Please send a website URL or choose 'No website'."
        )
        return self.WEBSITE

    async def website(self, update: Update, context: CallbackContext):
        """Saves a website link in the user data if one is sent"""
        try:
            self.logger.debug(
                f"website: Function successfully called with message {str(update.message.text)}"
            )

            if update.message.text == "no" and self.retry is True:
                response = (
                    "Okay, you keep your websites as they are.\n"
                    "Next: Do you want to upload a document? If yes, send it now. "
                    "If not, tap the button below or respond with 'no'."
                )
                await update.message.reply_text(
                    response, reply_markup=self.build_document_keyboard()
                )
                return self.DOCUMENT

            if update.message.text != "no" and self.retry is True:
                self.addWebsite(update.message.text)
                response = (
                    "Okay, do you have another website link? If yes, send it now. "
                    "If not, please tap 'No website' or respond with 'no'."
                )
                await update.message.reply_text(
                    response, reply_markup=self.build_website_keyboard()
                )
                return self.WEBSITE

            if update.message.text.lower() != "no":
                self.addWebsite(update.message.text)
                response = (
                    "Got it! 🌐\nDo you have another website link? "
                    "If yes, send it now. If not, please tap 'No website' or reply with 'no'."
                )
                await update.message.reply_text(
                    response, reply_markup=self.build_website_keyboard()
                )
                return self.WEBSITE

            response = (
                "Great! 📎 Do you have a *document* with information to include?\n"
                "If yes, upload the document now (PDF, DOCX, TXT). "
                "If not, tap the button below or send 'no'."
            )
            await update.message.reply_text(
                response, reply_markup=self.build_document_keyboard()
            )
            return self.DOCUMENT

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your link or 'no'."
            )
            self.logger.error(f"website: An exception occurred: {str(e)}")
            return self.WEBSITE

    # Step: Document

    async def document(self, update: Update, context: CallbackContext):
        """Saves a document in the 'documents' folder and the storage link in the user data if one is sent"""
        try:
            self.logger.debug("document: Function successfully called.")
            document = update.message.document
            if document:
                if document.mime_type not in self.VALID_MIME_TYPES:
                    await update.message.reply_text(
                        f"Unsupported file type: {document.mime_type}.\n"
                        f"Please upload a valid document (PDF, Word, TXT)."
                    )
                    return self.DOCUMENT

                base_dir = os.path.dirname(__file__)
                file_path = os.path.join(base_dir, "documents", document.file_name)
                self.logger.debug(f"document: File saved at: {file_path}")
                file_id = document.file_id
                file = await context.bot.get_file(file_id)
                await file.download_to_drive(file_path)

                match document.mime_type:
                    case "application/pdf":
                        self.addPDF(file_path)
                    case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        self.addDOCX(file_path)
                    case "text/plain":
                        self.addTxt(file_path)
                    case _:
                        await update.message.reply_text(
                            "Invalid file type, only acceptable file endings are: PDF, TXT and DOCX."
                        )
                        self.logger.warning(
                            f"document: Invalid file type sent: {document.mime_type}"
                        )
                        return self.DOCUMENT

            response = (
                "Do you have another document to upload? If yes, send it now.\n"
                "If not, tap the button below or reply with 'no'."
            )
            await update.message.reply_text(
                response, reply_markup=self.build_document_keyboard()
            )
            return self.DOCUMENT

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your document or 'no'."
            )
            self.logger.error(f"document: An exception occurred: {str(e)}")
            return self.DOCUMENT

    async def no_document_button(self, update: Update, context: CallbackContext):
        """Handles 'No document' button."""
        query = update.callback_query
        choice = query.data
        await query.answer()
        self.logger.debug(f"no_document_button: User chose {choice}")

        if choice == "no_document_btn":
            await query.edit_message_text(
                "Do you have a document with information to include?\n\n✅ Selected: No document"
            )
            response = "How long should the blog article be? Choose below 👇"
            await query.message.reply_text(
                response, reply_markup=self.build_length_keyboard()
            )
            return self.LENGTH

        await query.message.reply_text(
            "Please upload a document or choose 'No document'."
        )
        return self.DOCUMENT

    async def no_document(self, update: Update, context: CallbackContext):
        """Manages what happens when no document is sent"""
        try:
            self.logger.debug(
                f"no_document: called with message {str(update.message.text)}"
            )

            if update.message.text == "no" and self.retry is True:
                response = (
                    "Next, do you want to change your blog article *length*?\n"
                    "Choose one of the options below 👇"
                )
                await update.message.reply_text(
                    response, reply_markup=self.build_length_keyboard()
                )
                return self.LENGTH

            if update.message.text.lower() != "no":
                response = "Not valid, please send either a document or 'no'."
                await update.message.reply_text(response)
                return self.DOCUMENT

            response = "How long should the blog article be? Choose below 👇"
            await update.message.reply_text(
                response, reply_markup=self.build_length_keyboard()
            )
            return self.LENGTH

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your document or 'no'."
            )
            self.logger.error(f"no_document: An exception occurred: {str(e)}")
            return self.DOCUMENT

    # Step: Length

    async def length_button(self, update: Update, context: CallbackContext):
        """Handles length choice via InlineKeyboard."""
        query = update.callback_query
        choice = query.data
        await query.answer()
        self.logger.debug(f"length_button: User chose {choice}")
        context.user_data["length"] = choice

        await query.edit_message_text(
            f"How long should the blog article be?\n\n✅ Selected: {choice.capitalize()}"
        )

        response = "Great! What *language level* should it be?"
        await query.message.reply_text(
            response, reply_markup=self.build_level_keyboard()
        )
        return self.LANGUAGE_LEVEL

    async def length(self, update: Update, context: CallbackContext):
        """Fallback: saves length from free text."""
        try:
            self.logger.debug(f"length: called with message {str(update.message.text)}")
            context.user_data["length"] = update.message.text
            response = "Great! What *language level* should it be? (e.g. Beginner, Intermediate, Advanced)"
            await update.message.reply_text(
                response, reply_markup=self.build_level_keyboard()
            )
            return self.LANGUAGE_LEVEL

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your preferred article length."
            )
            self.logger.error(f"length: An exception occurred: {str(e)}")
            return self.LENGTH

    # Step: Language Level

    async def language_level_button(self, update: Update, context: CallbackContext):
        query = update.callback_query
        choice = query.data
        await query.answer()
        self.logger.debug(f"language_level_button: User chose {choice}")
        context.user_data["language_level"] = choice

        await query.edit_message_text(
            "What *language level* should it be?\n\n"
            f"✅ Selected: {choice.capitalize()}"
        )

        response = "Nice! 📊 What *information level* should it be?"
        await query.message.reply_text(
            response, reply_markup=self.build_info_keyboard()
        )
        return self.INFORMATION

    async def language_level(self, update: Update, context: CallbackContext):
        """Fallback: saves the configured language level in the user data"""
        try:
            self.logger.debug(
                f"language_level: called with message {str(update.message.text)}"
            )
            context.user_data["language_level"] = update.message.text
            response = "Great! What *information level* should it be? (e.g. High, Intermediate, Low)"
            await update.message.reply_text(
                response, reply_markup=self.build_info_keyboard()
            )
            return self.INFORMATION

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your preferred article language level."
            )
            self.logger.error(f"language_level: An exception occurred: {str(e)}")
            return self.LANGUAGE_LEVEL

    # Step: Information Level

    async def information_button(self, update: Update, context: CallbackContext):
        query = update.callback_query
        choice = query.data
        await query.answer()
        mapped = "medium" if choice == "medium_info" else choice

        self.logger.debug(f"information_button: User chose {mapped}")
        context.user_data["information"] = mapped

        await query.edit_message_text(
            "What *information level* should it be?\n\n"
            f"✅ Selected: {mapped.capitalize()}"
        )

        response = "Great! 🌍 What *language* should the article be in? (e.g. English, German, Spanish)"
        await query.message.reply_text(response)
        return self.LANGUAGE

    async def information(self, update: Update, context: CallbackContext):
        """Fallback: saves the configured information level in the user data"""
        try:
            self.logger.debug(
                f"information: called with message {str(update.message.text)}"
            )
            context.user_data["information"] = update.message.text
            response = "Great! 🌍 What *language* should it be? (e.g. English, German, Spanish)"
            await update.message.reply_text(response)
            return self.LANGUAGE

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your preferred article information level."
            )
            self.logger.error(f"information: An exception occurred: {str(e)}")
            return self.INFORMATION

    # Step: Language

    async def language(self, update: Update, context: CallbackContext):
        """Saves the configured language in the user data and asks for tone via buttons"""
        try:
            self.logger.debug(
                f"language: called with message {str(update.message.text)}"
            )
            context.user_data["language"] = update.message.text

            response = "Great! 🎨 What *tone* should it be?"
            await update.message.reply_text(
                response, reply_markup=self.build_tone_keyboard()
            )
            return self.TONE

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your preferred article language."
            )
            self.logger.error(f"language: An exception occurred: {str(e)}")
            return self.LANGUAGE

    # Step: Tone

    async def tone_button(self, update: Update, context: CallbackContext):
        query = update.callback_query
        choice = query.data
        await query.answer()
        self.logger.debug(f"tone_button: User chose {choice}")
        context.user_data["tone"] = choice

        await query.edit_message_text(
            "What *tone* should the article have?\n\n"
            f"✅ Selected: {choice.capitalize()}"
        )

        response = (
            "Great! ℹ️ Do you have any *additional information* to include?\n"
            "Choose one of the options below."
        )
        await query.message.reply_text(
            response, reply_markup=self.build_additional_keyboard()
        )
        return self.ADDITIONAL

    async def tone(self, update: Update, context: CallbackContext):
        """Fallback: saves the configured tone in the user data and asks for additional info"""
        try:
            self.logger.debug(f"tone: called with message {str(update.message.text)}")
            context.user_data["tone"] = update.message.text
            response = (
                "Great! ℹ️ Do you have any *additional information* you want to have included?\n"
                "If not, please respond with 'no'."
            )
            await update.message.reply_text(
                response, reply_markup=self.build_additional_keyboard()
            )
            return self.ADDITIONAL

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your preferred article tone."
            )
            self.logger.error(f"tone: An exception occurred: {str(e)}")
            return self.TONE

    # Step: Additional information

    async def additional_button_choice(self, update: Update, context: CallbackContext):
        """Handles 'No additional info' vs 'Add details' buttons."""
        query = update.callback_query
        choice = query.data
        await query.answer()
        self.logger.debug(f"additional_button_choice: User chose {choice}")

        if choice == "no_additional":
            await query.edit_message_text(
                "Do you have any additional information to include?\n\n"
                "✅ Selected: No additional info"
            )
            context.user_data["additional_information"] = ""
            user_data = context.user_data
            response = (
                "Thanks! Here's what I got:\n"
                f"- Topic or Task: {user_data.get('topic')}\n"
                f"- Length: {user_data.get('length')}\n"
                f"- Language Level: {user_data.get('language_level')}\n"
                f"- Information Level: {user_data.get('information')}\n"
                f"- Language: {user_data.get('language')}\n"
                f"- Tone: {user_data.get('tone')}\n"
                f"- Additional Information: (none)\n\n"
                "Choose whether you want to confirm or restart:"
            )
            await query.message.reply_text(
                response, reply_markup=self.build_confirm_keyboard()
            )
            return self.CONFIRM

        if choice == "add_additional":
            await query.edit_message_text(
                "Okay, please type any *additional information* you want to include."
            )
            return self.ADDITIONAL

        await query.message.reply_text(
            "Please choose an option for additional information."
        )
        return self.ADDITIONAL

    async def additional(self, update: Update, context: CallbackContext):
        """Handles free-text additional information (when user chose 'Add details')."""
        try:
            self.logger.debug(
                f"additional_information: called with message {str(update.message.text)}"
            )
            context.user_data["additional_information"] = ""
            if update.message.text.lower() != "no":
                context.user_data["additional_information"] = update.message.text

            user_data = context.user_data
            response = (
                "Thanks! Here's what I got:\n"
                f"- Topic or Task: {user_data.get('topic')}\n"
                f"- Length: {user_data.get('length')}\n"
                f"- Language Level: {user_data.get('language_level')}\n"
                f"- Information Level: {user_data.get('information')}\n"
                f"- Language: {user_data.get('language')}\n"
                f"- Tone: {user_data.get('tone')}\n"
                f"- Additional Information: {user_data.get('additional_information')}\n\n"
                "Choose whether you want to confirm or restart:"
            )
            await update.message.reply_text(
                response, reply_markup=self.build_confirm_keyboard()
            )
            return self.CONFIRM

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nPlease resend your additional information."
            )
            self.logger.error(f"additional: An exception occurred: {str(e)}")
            return self.ADDITIONAL

    # Step: Confirm

    async def confirm_button(self, update: Update, context: CallbackContext):
        """Handle Confirm/Restart buttons."""
        query = update.callback_query
        choice = query.data
        await query.answer()
        self.logger.debug(f"confirm_button: User chose {choice}")

        if choice == "confirm":
            await query.edit_message_text(
                query.message.text + "\n\n✅ You confirmed this configuration."
            )

            await query.message.reply_text("Processing your request...")

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
                await query.message.reply_text(str(result))
                self.logger.debug("confirm_button: Crew run successful.")
            except Exception as crew_error:
                self.logger.error(
                    f"confirm_button: Error during crew execution: {crew_error}",
                    exc_info=True,
                )
                await query.message.reply_text(
                    "❌ An error occurred during article generation. Please try again."
                )

            return ConversationHandler.END

        if choice == "restart":
            await query.edit_message_text(
                query.message.text + "\n\n🔁 You chose to restart the wizard."
            )
            self.retry = True
            response = (
                "Okay! You can change any answer. We'll go through the questions again.\n"
                "If you want to keep a previous answer, just reply with 'no' at that step.\n\n"
                "First: Do you want to base the article on a topic or a task?"
            )
            await query.message.reply_text(
                response, reply_markup=self.build_topic_or_task_keyboard()
            )
            return first_state()

        await query.message.reply_text(
            "Please choose whether you want to confirm or restart."
        )
        return self.CONFIRM

    async def confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Shows summary of all collected configuration values and handles final confirmation."""
        try:
            user_reply = update.message.text.strip().lower()

            if user_reply in ["yes", "y", "ja"]:
                self.logger.debug("confirm: User confirmed configuration.")

                await update.message.reply_text("Processing your request...")

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

                except Exception as crew_error:
                    self.logger.error(
                        f"confirm: Error during crew execution: {crew_error}",
                        exc_info=True,
                    )
                    await update.message.reply_text(
                        "❌ An error occurred during article generation. Please try again."
                    )

                return ConversationHandler.END

            elif user_reply in ["no", "n", "nein"]:
                self.logger.debug(
                    "confirm: User rejected configuration. Restarting questions."
                )

                await update.message.reply_text(
                    "Okay! You can change any answer. We'll go through the questions again.\n"
                    "If you want to keep a previous answer, just reply with 'no' at that step."
                )

                self.retry = True
                response = (
                    "First: Do you want to base the article on a topic or a task?"
                )
                await update.message.reply_text(
                    response, reply_markup=self.build_topic_or_task_keyboard()
                )
                return first_state()

            else:
                self.logger.debug("confirm: Invalid answer, prompting again.")

                await update.message.reply_text(
                    "Please respond with 'yes' to confirm or 'no' to restart.\n"
                    "Or use the buttons above."
                )
                return S.CONFIRM

        except Exception as e:
            self.logger.error(f"confirm: Unexpected error: {e}", exc_info=True)
            await update.message.reply_text(
                "An unexpected error occurred in the confirmation step."
            )
            return ConversationHandler.END

    # Cancel

    async def cancel(self, update: Update, context: CallbackContext):
        """The fallout function, leaves the conversation"""
        try:
            self.logger.debug(f"cancel: called with message {str(update.message.text)}")
            response = "Conversation canceled. Type /start to begin again."
            await update.message.reply_text(response)
            return ConversationHandler.END

        except Exception as e:
            await update.message.reply_text(
                f"An error occurred: {str(e)}. \nConversation canceled. Type /start to begin again."
            )
            self.logger.error(f"cancel: An exception occurred: {str(e)}")
            return ConversationHandler.END

    # Start bot

    def start_bot(self) -> None:
        """Builds and starts the Telegram bot with the conversation handler."""

        application = Application.builder().token(self.token).build()

        conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler("start_configuration", self.start_configuration),
            ],
            states={
                S.TOPIC_OR_TASK: [
                    CallbackQueryHandler(self.topic_or_task_button),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.topic_or_task,
                    ),
                ],
                S.TOPIC: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.topic,
                    )
                ],
                S.TASK: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.task,
                    )
                ],
                S.WEBSITE: [
                    CallbackQueryHandler(self.website_button),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.website,
                    ),
                ],
                S.DOCUMENT: [
                    CallbackQueryHandler(self.no_document_button),
                    MessageHandler(
                        filters.Document.ALL & ~filters.COMMAND,
                        self.document,
                    ),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.no_document,
                    ),
                ],
                S.LENGTH: [
                    CallbackQueryHandler(self.length_button),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.length,
                    ),
                ],
                S.LEVEL: [
                    CallbackQueryHandler(self.language_level_button),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.language_level,
                    ),
                ],
                S.INFO: [
                    CallbackQueryHandler(self.information_button),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.information,
                    ),
                ],
                S.LANGUAGE: [
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.language,
                    )
                ],
                S.TONE: [
                    CallbackQueryHandler(self.tone_button),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.tone,
                    ),
                ],
                S.ADDITIONAL: [
                    CallbackQueryHandler(self.additional_button_choice),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.additional,
                    ),
                ],
                S.CONFIRM: [
                    CallbackQueryHandler(self.confirm_button),
                    MessageHandler(
                        filters.TEXT & ~filters.COMMAND,
                        self.confirm,
                    ),
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel),
            ],
            name="blog_config_conversation",
        )

        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(CommandHandler("clear", self.clear))
        application.add_handler(CommandHandler("cancel", self.cancel))
        application.add_handler(conv_handler)
        application.run_polling()

    # RAG Tools

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
