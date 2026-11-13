import {
	StateGraph,
	StateSchema,
	ReducedValue,
	type GraphNode,
	START,
	END,
	Send,
} from "@langchain/langgraph";
import { z } from "zod/v4";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatGroq } from "@langchain/groq";
import { env } from "../config/env";
import { LangChainTracer } from "@langchain/core/tracers/tracer_langchain";
import { writeFile } from "node:fs/promises";

const TaskSchema = z.object({
	id: z.number(),
	title: z.string(),
	brief: z.string(),
});

const PlanSchema = z.object({
	blog_title: z.string(),
	tasks: z.array(TaskSchema),
});

const tracer = new LangChainTracer();

const model = new ChatGroq({
	model: env.LLM_MODEL,
	temperature: 0.2,
});

const MessagesState = new StateSchema({
	topic: z.string(),
	plan: PlanSchema,
	sections: new ReducedValue(
		z.array(z.string()).default(() => []),
		{ reducer: (curr, next) => curr.concat(next) },
	),
	final: z.string(),
});

const orchestrator: GraphNode<typeof MessagesState> = async (state) => {
	const topic = state.topic;
	const structuredModel = model.withStructuredOutput(PlanSchema);
	const conversation = [
		new SystemMessage(
			"Create a blog plan with 5-7 sections on the following topic.",
		),
		new HumanMessage(`Topic:${topic}`),
	];

	const response = await structuredModel.invoke(conversation, {
		callbacks: [tracer],
	});
	return { plan: response };
};

const fanout = (state: typeof MessagesState.State) => {
	return state.plan.tasks.map(
		(task) =>
			new Send("worker", { task: task, topic: state.topic, plan: state.plan }),
	);
};

const worker = async (state: {
	task: z.infer<typeof TaskSchema>;
	topic: string;
	plan: z.infer<typeof PlanSchema>;
}) => {
	const { task, topic, plan } = state;
	const blogTitle = plan.blog_title;
	const conversation = [
    	new SystemMessage(
			"Write one clean Markdown section."
		),
		new HumanMessage(`
      Blog:${blogTitle}\n
      Topic:${topic}\n
      Section:${task.title}\n
      Brief:${task.brief}\n
      Return only the section content in Markdown.`),
	];
  const response = (await model.invoke(conversation,{callbacks:[tracer]}));
  console.log(response);
  return { sections: [response.text] };
};

const reducer: GraphNode<typeof MessagesState> = async (state) => {
   // 1. Get the blog title
  const title = state.plan.blog_title;

  // 2. Combine all sections into one Markdown body
  const body = state.sections.join("\n\n").trim();

  // 3. Generate the final Markdown document
  const finalMd = `# ${title}\n\n${body}\n`;

  // 4. Save the Markdown document to a file
  const filename = title.toLowerCase().replace(/ /g, "_") + ".md";

  await writeFile(filename, finalMd, "utf-8");

  // 5. Return the updated graph state
  return { final: finalMd };
};

export const agent = new StateGraph(MessagesState)
	.addNode("orchestrator", orchestrator)
	.addNode("worker", worker)
	.addNode("reducer", reducer)

	.addEdge(START, "orchestrator")
  .addConditionalEdges('orchestrator',fanout)
	.addEdge("worker", "reducer")
	.addEdge("reducer", END)
	.compile();

// const result = await agent.invoke({
// 	topic: "Write a blog on Self Attention",
// });

// console.log(result.final);

// const drawableGraph = await agent.getGraphAsync();
// console.log(drawableGraph.drawMermaid());
