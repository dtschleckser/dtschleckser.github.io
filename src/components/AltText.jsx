import { HoverCard, Text } from '@mantine/core'

function AltText({text, alt, link}) {
    return (
    <HoverCard width={280} shadow="md">
        <HoverCard.Target>
            {link ? 
            <a href={link} target="_blank">{" "}{text}{" "}</a> :
            <a href="#">{" "}{text}{" "}</a>
            }
        </HoverCard.Target>
        <HoverCard.Dropdown>
        <Text size="sm">
            {alt}{' '}
            {link ? <a href={link}>🔗</a> : ""}
        </Text>
        </HoverCard.Dropdown>
    </HoverCard>
    )
}

export default AltText